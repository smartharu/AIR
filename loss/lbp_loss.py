import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

_DISABLE_COMPILE = sys.platform != "linux" or os.getenv("NUNIF_DISABLE_COMPILE", False)


def compile(*args, **kwargs):
    if _DISABLE_COMPILE:
        return args[0]
    else:
        return torch.compile(*args, **kwargs)


def conditional_compile(env_name):
    def decorator(*args, **kwargs):
        env_names = env_name if isinstance(env_name, (list, tuple)) else [env_name]
        cond = any([int(os.getenv(name, "0")) for name in env_names])
        if not cond or _DISABLE_COMPILE:
            return args[0]
        else:
            return torch.compile(*args, **kwargs)
    return decorator



def generate_lbcnn_filters(size, sparcity=0.9, seed=71):
    rng_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        filters = torch.bernoulli(torch.torch.full(size, 0.5)).mul_(2).add(-1)
        filters[torch.rand(filters.shape) > sparcity] = 0
    finally:
        torch.random.set_rng_state(rng_state)
    # print(filters)

    return filters

def generate_lbp_kernel(in_channels, out_channels, kernel_size=3, seed=71):
    kernel = generate_lbcnn_filters((out_channels, in_channels, kernel_size, kernel_size), seed=seed)
    # [0] = identity filter
    kernel[0] = 0
    kernel[0, :, kernel_size // 2, kernel_size // 2] = 0.5 * kernel_size ** 2
    return kernel


def charbonnier_loss(input, target, reduction="mean", eps=1.0e-3):
    loss = torch.sqrt(((input - target) ** 2) + eps ** 2)
    if reduction is None or reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    else:
        return loss.sum()


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3, reduction="mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, input, target):
        return charbonnier_loss(input, target, eps=self.eps, reduction=self.reduction)


def clamp_loss(input, target, loss_function, min_value, max_value, eta=0.001):
    noclip_loss = loss_function(input, target)
    clip_loss = loss_function(torch.clamp(input, min_value, max_value),
                              torch.clamp(target, min_value, max_value))
    return clip_loss + noclip_loss * eta


def clamp_l1_loss(input, target, loss_function, min_value, max_value, eta=0.001):
    noclip_loss = F.l1_loss(input, target)
    clip_loss = loss_function(torch.clamp(input, min_value, max_value),
                              torch.clamp(target, min_value, max_value))
    return clip_loss + noclip_loss * eta


class ClampLoss(nn.Module):
    """ Wrapper Module for `(clamp(input, 0, 1) - clamp(target, 0, 1))`
    """
    def __init__(self, module, min_value=0, max_value=1, eta=0.001):
        super().__init__()
        self.module = module
        self.min_value = min_value
        self.max_value = max_value
        self.eta = eta

    def forward(self, input, target):
        return clamp_loss(input, target, self.module,
                          min_value=self.min_value, max_value=self.max_value, eta=self.eta)


def channel_weighted_loss(input, target, loss_func, weight):
    return sum([loss_func(input[:, i:i + 1, :, :], target[:, i:i + 1, :, :]) * w
                for i, w in enumerate(weight)])


class ChannelWeightedLoss(nn.Module):
    """ Wrapper Module for channel weight
    """
    def __init__(self, module, weight):
        super().__init__()
        self.module = module
        self.weight = weight

    def forward(self, input, target):
        b, ch, *_ = input.shape
        assert (ch == len(self.weight))
        return channel_weighted_loss(input, target, self.module, self.weight)


LUMINANCE_WEIGHT = [0.29891, 0.58661, 0.11448]


class LuminanceWeightedLoss(ChannelWeightedLoss):
    def __init__(self, module):
        super().__init__(module, weight=LUMINANCE_WEIGHT)


class AverageWeightedLoss(ChannelWeightedLoss):
    def __init__(self, module, in_channels=3):
        weight = [1.0 / in_channels] * in_channels
        super().__init__(module, weight=weight)

class LBPLoss(nn.Module):
    def __init__(self, in_channels, out_channels=64, kernel_size=3, loss=None, seed=71):
        super().__init__()
        self.groups = in_channels
        self.register_buffer(
            "kernel",
            generate_lbp_kernel(in_channels, out_channels - out_channels % in_channels,
                                kernel_size, seed=seed))
        if loss is None:
            self.loss = CharbonnierLoss()
        else:
            self.loss = loss

    def conv(self, x):
        return F.conv2d(x, weight=self.kernel, bias=None, stride=1, padding=0, groups=self.groups)

    @conditional_compile("NUNIF_TRAIN")
    def forward(self, input, target):
        b, ch, *_ = input.shape
        return self.loss(self.conv(input), self.conv(target))


class YLBP(nn.Module):
    def __init__(self, kernel_size=3, out_channels=64):
        super().__init__()
        self.eta = 0.001
        self.loss = CharbonnierLoss()
        self.register_buffer("kernel", generate_lbp_kernel(1, out_channels, kernel_size))

    def conv(self, x):
        return F.conv2d(x, weight=self.kernel, bias=None, stride=1, padding=0)

    @conditional_compile("NUNIF_TRAIN")
    def forward(self, input, target):
        B, C, *_ = input.shape
        target = torch.clamp(target, 0, 1)
        target_feat = [self.conv(target[:, i:i + 1, :, :]) for i in range(C)]
        nonclip_loss = sum([self.loss(self.conv(input[:, i:i + 1, :, :]), target_feat[i]) * w
                            for i, w in enumerate(LUMINANCE_WEIGHT)])
        clip_loss = sum([self.loss(self.conv(torch.clamp(input[:, i:i + 1, :, :], 0, 1)), target_feat[i]) * w
                         for i, w in enumerate(LUMINANCE_WEIGHT)])
        return clip_loss + nonclip_loss * self.eta


def RGBLBP(kernel_size=3):
    return ClampLoss(AverageWeightedLoss(LBPLoss(in_channels=1, kernel_size=kernel_size),
                                         in_channels=3))


class YL1LBP(nn.Module):
    def __init__(self, kernel_size=5, weight=0.4):
        super().__init__()
        self.lbp = YLBP(kernel_size=kernel_size)
        self.l1 = ClampLoss(LuminanceWeightedLoss(torch.nn.L1Loss()))
        self.weight = weight

    @conditional_compile("NUNIF_TRAIN")
    def forward(self, input, target):
        lbp_loss = self.lbp(input, target)
        l1_loss = self.l1(input, target)
        return l1_loss + lbp_loss * self.weight


class L1LBP(nn.Module):
    def __init__(self, kernel_size=5, weight=0.4):
        super().__init__()
        self.lbp = RGBLBP(kernel_size=kernel_size)
        self.l1 = ClampLoss(AverageWeightedLoss(torch.nn.L1Loss(), in_channels=3))
        self.weight = weight

    @conditional_compile("NUNIF_TRAIN")
    def forward(self, input, target):
        lbp_loss = self.lbp(input, target)
        l1_loss = self.l1(input, target)
        return l1_loss + lbp_loss * self.weight


def _check_gradient_norm():
    l1_loss = nn.L1Loss()
    lbp_loss = RGBLBP()

    x1 = torch.ones((1, 3, 32, 32), requires_grad=True) / 2.
    x2 = torch.ones((1, 3, 32, 32), requires_grad=True) / 2.
    x2 = x2 + torch.randn(x2.shape) * 0.01

    loss1 = l1_loss(x1, x2)
    loss2 = lbp_loss(x1, x2)

    grad1 = torch.autograd.grad(loss1, x2, retain_graph=True)[0]
    grad2 = torch.autograd.grad(loss2, x2, retain_graph=True)[0]
    norm1 = torch.norm(grad1, p=2)
    norm2 = torch.norm(grad2, p=2)

    # norm1 / norm2 = around 0.41
    print(norm1, norm2, norm1 / norm2)


def _test_clamp_input_only():
    import time

    lbp_old = ClampLoss(LuminanceWeightedLoss(LBPLoss(in_channels=1, kernel_size=3))).cuda()
    lbp_new = YLBP().cuda()

    torch.manual_seed(71)
    x = torch.randn((4, 3, 256, 256)) / 2 + 0.5
    y = torch.clamp(x + (torch.randn((4, 3, 256, 256)) / 10), 0, 1)
    x = x.cuda()
    y = y.cuda()
    print("diff", (lbp_old(x, y) - lbp_new(x, y)).abs())

    N = 100
    t = time.perf_counter()
    for _ in range(N):
        lbp_old(x, y)
    torch.cuda.synchronize()
    print("ylbp_old", time.perf_counter() - t)
    t = time.perf_counter()
    for _ in range(N):
        lbp_new(x, y)
    torch.cuda.synchronize()
    print("ylbp_new", time.perf_counter() - t)


if __name__ == "__main__":
    # _check_gradient_norm()
    _test_clamp_input_only()
