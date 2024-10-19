import torch
import torchvision.transforms.functional as TTF
import torch.nn.functional as TF
from data.wand_io import to_wand_image, to_tensor
from data.color import rgb_to_yuv, yuv_to_rgb
from scipy import special
import numpy as np
import random
from typing import Tuple


def resize(img: torch.Tensor, w: int, h: int, kernel: str = "cubic") -> torch.Tensor:
    with to_wand_image(img) as im:
        im.resize(width=w, height=h, filter=kernel)
        return to_tensor(im)


def filter2D(img: torch.Tensor, kernel_size: int, kernel: torch.Tensor) -> torch.Tensor:
    dim = len(img.shape)

    if dim == 2:
        img = img.unsqueeze(0).unsqueeze(0)

    elif dim == 3:
        img = img.unsqueeze(0)

    b, c, h, w = img.shape

    img_pad = TF.pad(img, (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2), mode="reflect")

    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(c, 1, 1, 1)

    out = TF.conv2d(img_pad, weight=kernel, bias=None, stride=1, groups=c)

    if dim == 2:
        out = out.squeeze(0).squeeze(0)
    elif dim == 3:
        out = out.squeeze(0)

    return out


def unsharp_mask(img: torch.Tensor, blur: str = "gaussian", kernel_size: int = 3, sigma: float = 2.0, gray: bool = True,
                 radius: int = 1, amount: float = 1.0) -> torch.Tensor:
    if blur == "gaussian":
        nr = gaussian_blur(img, kernel_size, sigma).clamp(0, 1)
    elif blur == "box":
        nr = box_blur(img, radius)
    else:
        raise ValueError("未知blur")

    makediff = (img - nr) * amount
    mergediff = (img + makediff).clamp(0, 1)

    if gray:
        img_yuv = rgb_to_yuv(img)
        img_u = img_yuv[..., 1, :, :]
        img_v = img_yuv[..., 2, :, :]

        sharp_yuv = rgb_to_yuv(mergediff)
        sharp_y = sharp_yuv[..., 0, :, :]

        res = yuv_to_rgb(torch.stack([sharp_y, img_u, img_v], -3))
    else:
        res = mergediff

    return res.clamp(0, 1)


def gaussian_blur(img, ksize=5, sigma=0.5):
    # gaussian kernel

    n = (ksize - 1.0) / 2.0
    x = torch.arange(-n, n + 1, dtype=torch.float32).reshape(ksize, 1)
    y = torch.arange(-n, n + 1, dtype=torch.float32).reshape(1, ksize)
    h = torch.exp(torch.div(-((x * x) + (y * y)), 2 * sigma * sigma))
    h = torch.div(h, h.sum())

    out = filter2D(img, ksize, h)

    return out.clamp(0, 1)


def box_blur(img: torch.Tensor, radius: int = 1) -> torch.Tensor:
    ksize = 2 * radius + 1

    k = torch.ones((ksize, ksize), dtype=torch.float32)
    k = k / k.sum()

    out = filter2D(img, ksize, k)
    return out.clamp(0, 1)


def mod_blur(img: torch.Tensor) -> torch.Tensor:
    ksize = 3

    parms = random.sample(range(1, 10), k=3)
    parms.sort()
    parm1 = parms[0]
    parm2 = parms[1]
    parm3 = parms[2]

    k = torch.tensor([[parm1, parm2, parm1], [parm2, parm3, parm2], [parm1, parm2, parm1]], dtype=torch.float32)
    k = k / k.sum()

    out = filter2D(img, ksize, k)
    return out.clamp(0, 1)


def lanczos_filter(img: torch.Tensor, ksize: int, sigma: float) -> torch.Tensor:
    n = (ksize - 1.0) / 2.0
    x = torch.arange(-n, n + 1, dtype=torch.float32).reshape(ksize, 1)
    y = torch.arange(-n, n + 1, dtype=torch.float32).reshape(1, ksize)

    d = torch.sqrt((x * x) + (y * y))
    h = torch.sinc(d) * torch.sinc(d / sigma)
    h = torch.div(h, h.sum())

    out = filter2D(img, ksize, h)
    return out.clamp(0, 1)


def sinc_filter(img: torch.Tensor, ksize: int, cutoff: float, pad_to: int = 0) -> torch.Tensor:
    kernel = np.fromfunction(
        lambda x, y: cutoff * special.j1(cutoff * np.sqrt(
            (x - (ksize - 1) / 2) ** 2 + (y - (ksize - 1) / 2) ** 2)) / (2 * np.pi * np.sqrt(
            (x - (ksize - 1) / 2) ** 2 + (y - (ksize - 1) / 2) ** 2)), [ksize, ksize])
    kernel[(ksize - 1) // 2, (ksize - 1) // 2] = cutoff ** 2 / (4 * np.pi)
    kernel = kernel / np.sum(kernel)
    if pad_to > ksize:
        pad_size = (pad_to - ksize) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

    k = torch.from_numpy(kernel).float()
    out = filter2D(img, ksize, k)
    return out.clamp(0, 1)


def laplacian_sharpen(img: torch.Tensor, gray: bool = True) -> torch.Tensor:
    if random.uniform(0, 1) < 0.5:
        k = torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=torch.float32)
    else:
        k = torch.tensor([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=torch.float32)

    if gray:
        img_yuv = rgb_to_yuv(img)
        y = img_yuv[..., 0, :, :]
        u = img_yuv[..., 1, :, :]
        v = img_yuv[..., 2, :, :]
        y = filter2D(y, 3, k)
        out = yuv_to_rgb(torch.stack([y, u, v], -3))
    else:
        out = filter2D(img, 3, k)

    return out.clamp(0, 1)


def sharpen(img: torch.Tensor, amount: float = 1.0, gray: bool = True) -> torch.Tensor:
    kx = torch.tensor([(1 - 2 ** amount) / 2, 2 ** amount, (1 - 2 ** amount) / 2], dtype=torch.float32).reshape(3, 1)
    ky = torch.tensor([(1 - 2 ** amount) / 2, 2 ** amount, (1 - 2 ** amount) / 2], dtype=torch.float32).reshape(1, 3)
    k = kx * ky

    if gray:
        img_yuv = rgb_to_yuv(img)
        y = img_yuv[..., 0, :, :]
        u = img_yuv[..., 1, :, :]
        v = img_yuv[..., 2, :, :]
        y = filter2D(y, 3, k)
        out = yuv_to_rgb(torch.stack([y, u, v], -3))
    else:
        out = filter2D(img, 3, k)

    return out.clamp(0, 1)


class RandomGray:
    def __init__(self, prob: float = 0.3) -> None:
        self.prob = prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.uniform(0, 1) > self.prob:
            return x
        return TTF.rgb_to_grayscale(x, 3)


class RandomLanczosFilter:
    def __init__(self, prob: float = 0.3, kernel_size: Tuple[int, int] | int = 5,
                 sigma: Tuple[float, float] | float = (2.0, 5.0)) -> None:
        self.prob = prob
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.uniform(0, 1) > self.prob:
            return x

        if isinstance(self.kernel_size, int):
            if (self.kernel_size & 1) == 0:
                raise ValueError("kernel_size 应该为奇数")
            kernel_size = self.kernel_size
        elif isinstance(self.kernel_size, tuple):
            if (self.kernel_size[0] & 1) == 0 or (self.kernel_size[1] & 1) == 0:
                raise ValueError("kernel_size 应该为奇数")
            kernel_size = random.choice(list(range(self.kernel_size[0], self.kernel_size[1] + 1, 2)))
        else:
            raise TypeError("kernel_size 应该为int或者(int,int)")

        if isinstance(self.sigma, float):
            if self.sigma < 0:
                raise ValueError("sigma 必须大于0")
            sigma = self.sigma
        elif isinstance(self.sigma, tuple):
            if self.sigma[0] < 0 or self.sigma[1] < 0:
                raise ValueError("sigma 必须大于0")
            sigma = random.uniform(self.sigma[0], self.sigma[1])
        else:
            raise TypeError("sigma 应该为float或者(float,float)")

        ret = lanczos_filter(x, kernel_size, sigma)
        return ret


'''
def repair(flt, src):
    if len(flt.shape) == 2:
        flt = flt.unsqueeze(0)

    if len(src.shape) == 2:
        src = src.unsqueeze(0)

    channel, height, width = src.shape

    src_pad = F.pad(src, (1, 1, 1, 1), mode="constant")

    for i, c in enumerate(src_pad):
        m1 = c[0:height, 0:width]
        m2 = c[0:height, 1:width + 1]
        m3 = c[0:height, 2:width + 2]
        m4 = c[1:height + 1, 0:width]
        m5 = c[1:height + 1, 1:width + 1]
        m6 = c[1:height + 1, 2:width + 2]
        m7 = c[2:height + 2, 0:width]
        m8 = c[2:height + 2, 1:width + 1]
        m9 = c[2:height + 2, 2:width + 2]
        mstack = torch.stack([m1, m2, m3, m4, m5, m6, m7, m8, m9], dim=0)
        stack_max, _ = torch.max(mstack, dim=0)
        stack_min, _ = torch.min(mstack, dim=0)
        cp_max, _ = torch.max(torch.stack([flt[i], stack_min], dim=0), dim=0)
        cp_res, _ = torch.min(torch.stack([cp_max, stack_max], dim=0), dim=0)
        flt[i] = cp_res

    return flt
'''


class RandomSincFilter:
    def __init__(self, prob: float = 0.3, kernel_size: Tuple[int, int] | int = 5,
                 sigma: Tuple[float, float] | float = (2.0, 5.0)) -> None:
        self.prob = prob
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.uniform(0, 1) > self.prob:
            return x

        if isinstance(self.kernel_size, int):
            if (self.kernel_size & 1) == 0:
                raise ValueError("kernel_size 应该为奇数")
            kernel_size = self.kernel_size
        elif isinstance(self.kernel_size, tuple):
            if (self.kernel_size[0] & 1) == 0 or (self.kernel_size[1] & 1) == 0:
                raise ValueError("kernel_size 应该为奇数")
            kernel_size = random.choice(list(range(self.kernel_size[0], self.kernel_size[1] + 1, 2)))
        else:
            raise TypeError("kernel_size 应该为int或者(int,int)")

        if isinstance(self.sigma, float):
            if self.sigma < 0:
                raise ValueError("sigma 必须大于0")
            sigma = self.sigma
        elif isinstance(self.sigma, tuple):
            if self.sigma[0] < 0 or self.sigma[1] < 0:
                raise ValueError("sigma 必须大于0")
            sigma = random.uniform(self.sigma[0], self.sigma[1])
        else:
            raise TypeError("sigma 应该为float或者(float,float)")

        ret = sinc_filter(x, kernel_size, sigma, 0)

        return ret


class RandomBoxBlur:
    def __init__(self, prob: float = 0.3, radius: Tuple[int, int] | int = 1) -> None:
        self.prob = prob
        self.radius = radius

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.uniform(0, 1) > self.prob:
            return x

        if isinstance(self.radius, int):
            if self.radius <= 0:
                raise ValueError("radius 必须大于0")
            radius = self.radius
        elif isinstance(self.radius, tuple):
            if self.radius[0] <= 0 or self.radius[1] <= 0:
                raise ValueError("radius 必须大于0")
            radius = random.randint(self.radius[0], self.radius[1])
        else:
            raise TypeError("radius 应该为int或者(int,int)")

        if random.uniform(0, 1) < 0.3:
            ret = box_blur(x, radius)
        else:
            ret = mod_blur(x)

        return ret


class RandomGaussianBlur:
    def __init__(self, prob: float = 0.3, kernel_size: Tuple[int, int] | int = 5,
                 sigma: Tuple[float, float] | float = (0.1, 2.0)) -> None:
        self.prob = prob
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.uniform(0, 1) > self.prob:
            return x

        if isinstance(self.kernel_size, int):
            if (self.kernel_size & 1) == 0:
                raise ValueError("kernel_size 应该为奇数")
            kernel_size = self.kernel_size
        elif isinstance(self.kernel_size, tuple):
            if (self.kernel_size[0] & 1) == 0 or (self.kernel_size[1] & 1) == 0:
                raise ValueError("kernel_size 应该为奇数")
            kernel_size = random.choice(list(range(self.kernel_size[0], self.kernel_size[1] + 1, 2)))
        else:
            raise TypeError("kernel_size 应该为int或者(int,int)")

        if isinstance(self.sigma, float):
            if self.sigma < 0:
                raise ValueError("sigma 必须大于0")
            sigma = self.sigma
        elif isinstance(self.sigma, tuple):
            if self.sigma[0] < 0 or self.sigma[1] < 0:
                raise ValueError("sigma 必须大于0")
            sigma = random.uniform(self.sigma[0], self.sigma[1])
        else:
            raise TypeError("sigma 应该为float或者(float,float)")

        ret = gaussian_blur(x, kernel_size, sigma).clamp(0, 1)

        return ret


class RandomBlur:
    def __init__(self, prob: float = 0.3, kernel_size: Tuple[int, int] | int = 5,
                 sigma: Tuple[float, float] | float = (0.1, 1.0),
                 radius: Tuple[int, int] | int = 1) -> None:
        self.prob = prob
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.radius = radius

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.uniform(0, 1) > self.prob:
            return x

        if isinstance(self.kernel_size, int):
            if (self.kernel_size & 1) == 0:
                raise ValueError("kernel_size 应该为奇数")
            kernel_size = self.kernel_size
        elif isinstance(self.kernel_size, tuple):
            if (self.kernel_size[0] & 1) == 0 or (self.kernel_size[1] & 1) == 0:
                raise ValueError("kernel_size 应该为奇数")
            kernel_size = random.choice(list(range(self.kernel_size[0], self.kernel_size[1] + 1, 2)))
        else:
            raise TypeError("kernel_size 应该为int或者(int,int)")

        if isinstance(self.sigma, float):
            if self.sigma < 0:
                raise ValueError("sigma 必须大于0")
            sigma = self.sigma
        elif isinstance(self.sigma, tuple):
            if self.sigma[0] < 0 or self.sigma[1] < 0:
                raise ValueError("sigma 必须大于0")
            sigma = random.uniform(self.sigma[0], self.sigma[1])
        else:
            raise TypeError("sigma 应该为float或者(float,float)")

        if isinstance(self.radius, int):
            if self.radius <= 0:
                raise ValueError("radius 必须大于0")
            radius = self.radius
        elif isinstance(self.radius, tuple):
            if self.radius[0] <= 0 or self.radius[1] <= 0:
                raise ValueError("radius 必须大于0")
            radius = random.randint(self.radius[0], self.radius[1])
        else:
            raise TypeError("radius 应该为int或者(int,int)")
        if random.uniform(0, 1) < 0.5:
            if random.uniform(0, 1) < 0.5:
                ret = box_blur(x, radius)
            else:
                ret = mod_blur(x)
        else:
            ret = gaussian_blur(x, kernel_size, sigma).clamp(0, 1)

        return ret

class RandomSharpen:
    def __init__(self, prob: float = 0.3, sigma: Tuple[float, float]|float = (2.0, 5.0),
                 radius: Tuple[int, int]|int = (1, 2), gray_prob: float = 0.5) -> None:
        self.prob = prob
        self.gray_prob = gray_prob
        self.sigma = sigma
        self.radius = radius

        self.kernel_sizes = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
        self.kernel_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        self.blur_type = ["gaussian", "box"]

        self.sharpen_type = ["usm", "laplacian", "avs"]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.uniform(0, 1) > self.prob:
            return x

        if random.uniform(0, 1) > self.gray_prob:
            gray = False
        else:
            gray = True

        sp_type = random.choice(self.sharpen_type)

        if sp_type == "usm":
            sigma = random.uniform(self.sigma[0], self.sigma[1])

            radius = random.randint(self.radius[0], self.radius[1])

            ksize = random.choices(self.kernel_sizes, weights=self.kernel_weights, k=1)[0]

            amount = 1
            if random.uniform(0, 1) < 0.3:
                amount = random.uniform(0.5, 2.0)

            blur_type = random.choice(self.blur_type)

            sharp = unsharp_mask(x, blur_type, ksize, sigma, gray, radius, amount)
        elif sp_type == "laplacian":
            sharp = laplacian_sharpen(x, gray)
        else:
            amount = random.uniform(0, 1)
            sharp = sharpen(x, amount, gray)

        return sharp