import lpips
from torch import nn

class LPIPSWith(nn.Module):
    def __init__(self, weight=1.):
        super().__init__()
        self.base_loss = nn.L1Loss()
        self.weight = weight
        self.lpips = lpips.LPIPS(net='vgg').eval()
        # This needed because LPIPS has duplicate parameter references problem
        self.lpips.requires_grad_(False)

    def forward(self, input, target):
        base_loss = self.base_loss(input, target)
        lpips_loss = self.lpips(input, target, normalize=True).mean()
        return base_loss + lpips_loss * self.weight
