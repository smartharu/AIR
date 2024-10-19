import torch
from torch import nn
from collections import OrderedDict
import math
import torch.nn.functional as F
from torch.ao.nn.quantized.functional import upsample


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


class DownSample(nn.Module):
    def __init__(self, down_scale: int = 2, downsample: str = "strideconv", kernel_size=3, stride: int = 1,
                 in_channels: int = 64, out_channels: int = 64, bias: bool = True):
        super(DownSample, self).__init__()

        if downsample == "strideconv":
            stride = down_scale
            kernel_size = 2
            self.down = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                  stride=stride, padding=0, bias=bias)
        elif downsample == "pixelunshuffle":
            self.down = nn.Sequential(
                nn.PixelUnshuffle(2),
                nn.Conv2d(in_channels=in_channels * 2 * 2, out_channels=out_channels, kernel_size=kernel_size,
                          stride=stride, padding=kernel_size // 2, bias=bias))
        else:
            raise TypeError("unsupported downsample")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        return x


class UpSample(nn.Module):
    def __init__(self, up_scale: int = 2, upsample: str = "upconv", kernel_size=3, stride: int = 1,
                 in_channels: int = 64, out_channels: int = 64, bias: bool = True):
        super(UpSample, self).__init__()

        if upsample == "convtranspose":
            stride = up_scale
            kernel_size = 2
            self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=0, bias=bias)
        elif upsample == "pixelshuffle":
            self.up = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels * 2 * 2, kernel_size=kernel_size,
                          stride=stride, padding=kernel_size // 2, bias=bias),
                nn.PixelShuffle(2))
        elif upsample == "upconv":
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                          padding=kernel_size // 2, bias=bias))
        else:
            raise TypeError("unsupported upsample")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return x


class UpSamplePixelShuffle(nn.Module):
    def __init__(self, scale: int = 2, in_channels: int = 64, out_channels: int = 64, bias: bool = True):
        super(UpSamplePixelShuffle, self).__init__()

        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(in_channels, 4 * in_channels, 3, 1, 1, bias=bias))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(in_channels, 9 * in_channels, 3, 1, 1, bias=bias))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')

        self.conv_before_upsample = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                                                  nn.LeakyReLU(0.2, inplace=True))

        self.upsample = nn.Sequential(*m)

        self.conv_last = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_before_upsample(x)
        x = self.upsample(x)
        x = self.conv_last(x)
        return x


class UpSampleNearestConv(nn.Module):
    def __init__(self, scale: int = 2, in_channels: int = 64, out_channels: int = 64, bias: bool = True):
        super(UpSampleNearestConv, self).__init__()
        self.scale = scale
        self.conv_before_upsample = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=bias),
                                                  nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv_up1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=bias)
        if self.scale == 4:
            self.conv_up2 = nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=bias)
        self.conv_hr = nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=bias)
        self.conv_last = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=bias)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_before_upsample(x)
        x = self.leaky_relu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
        if self.scale == 4:
            x = self.leaky_relu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
        x = self.conv_last(self.leaky_relu(self.conv_hr(x)))
        return x


class CALayer(nn.Module):
    def __init__(self, channel: int = 64, reduction: int = 1, bias: bool = True):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_fc = nn.Sequential(
            nn.Conv2d(channel, math.floor(channel / reduction), 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(math.floor(channel / reduction), channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv_fc(y)
        return x * y


class ResidualBlock(nn.Module):
    def __init__(self, channels: int = 64, kernel_size: int = 3, stride: int = 1, padding: int = 1, bias: bool = True):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size, stride, padding, bias=bias)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, stride, padding, bias=bias)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv2(self.leaky_relu(self.conv1(x)))
        return y + x


class ResidualChannelAttentionBlock(nn.Module):
    def __init__(self, channels: int = 64, kernel_size: int = 3, stride: int = 1, padding: int = 1, bias: bool = True,
                 reduction: int = 1):
        super(ResidualChannelAttentionBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size, stride, padding, bias=bias)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, stride, padding, bias=bias)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.channel_attention = CALayer(channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv2(self.leaky_relu(self.conv1(x)))
        y = self.channel_attention(y)
        return y + x


class ResidualChannelAttentionGroup(nn.Module):
    def __init__(self, channels: int = 64, kernel_size: int = 3, stride: int = 1, padding: int = 1, bias: bool = True,
                 reduction: int = 1, blocks: int = 2):
        super(ResidualChannelAttentionGroup, self).__init__()

        self.rcab = nn.Sequential(*[
            ResidualChannelAttentionBlock(channels, kernel_size, stride, padding, bias, reduction) for _ in
            range(blocks)])

    def forward(self, x):
        x = self.rcab(x)
        return x
