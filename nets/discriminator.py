from torch.nn.utils.parametrizations import spectral_norm
from torch import nn
import torch
from nets.model import Model

class SNSEBlock(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 8, bias: bool = True):
        super().__init__()

        self.attention = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                       spectral_norm(
                                           nn.Conv2d(in_channels, in_channels // reduction, 1, 1, 0, bias=bias)),
                                       nn.ReLU(inplace=True),
                                       spectral_norm(
                                           nn.Conv2d(in_channels // reduction, in_channels, 1, 1, 0, bias=bias)),

                                       nn.Sigmoid())

    def forward(self, x: torch.Tensor):
        y = self.attention(x)
        return y * x

class UNetDiscriminator(Model):
    def __init__(self, in_channels=3, num_feats=64):
        super().__init__()
        norm = spectral_norm

        self.head = nn.Conv2d(in_channels, num_feats, kernel_size=3, stride=1, padding=1)

        self.down1 = nn.Sequential(norm(nn.Conv2d(num_feats, num_feats * 2, 4, 2, 1, bias=False)),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   SNSEBlock(in_channels=num_feats * 2))

        self.down2 = nn.Sequential(norm(nn.Conv2d(num_feats * 2, num_feats * 4, 4, 2, 1, bias=False)),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   SNSEBlock(in_channels=num_feats * 4))

        self.down3 = nn.Sequential(norm(nn.Conv2d(num_feats * 4, num_feats * 8, 4, 2, 1, bias=False)),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   SNSEBlock(in_channels=num_feats * 8))

        self.up3 = nn.Sequential(norm(nn.Conv2d(num_feats * 8, num_feats * 4 * 4, 3, 1, 1, bias=False)),
                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                 nn.PixelShuffle(2),
                                 SNSEBlock(in_channels=num_feats * 4))

        self.up2 = nn.Sequential(norm(nn.Conv2d(num_feats * 4, num_feats * 2 * 4, 3, 1, 1, bias=False)),
                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                 nn.PixelShuffle(2),
                                 SNSEBlock(in_channels=num_feats * 2))

        self.up1 = nn.Sequential(norm(nn.Conv2d(num_feats * 2, num_feats * 4, 3, 1, 1, bias=False)),
                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                 nn.PixelShuffle(2),
                                 SNSEBlock(in_channels=num_feats))

        self.tail = nn.Conv2d(num_feats, 1, 3, 1, 1)

    def get_model_name(self):
        return "UNetDiscriminator"

    def forward(self, x: torch.Tensor):
        x1 = self.head(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up3(x4)
        x = self.up2(x + x3)
        x = self.up1(x + x2)
        x = self.tail(x + x1)

        return x


if __name__ == "__main__":
    x = torch.randn((1, 3, 128, 128))
    net = UNetDiscriminator()
    x = net(x)

    print(x.shape)
