import torch
from torch import nn as nn
from torch.nn import functional as F
from math import sqrt
import math


class SConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(SConvBlock, self).__init__()

        self.dconv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, groups=in_channels)
        self.pconv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.dconv(x)
        x = self.pconv(x)
        return x


class PatchDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PatchDown, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class PatchUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PatchUp, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, num_feat):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // 4, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // 4, num_feat, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.attention(x)
        return x * y


class SptialAttention(nn.Module):
    def __init__(self, num_feat):
        super(SptialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, num_feat // 4, 1, 1, 0)
        self.conv2 = nn.Conv2d(num_feat // 4, num_feat // 4, 3, 1, 1, 1, num_feat // 4)
        self.conv3 = nn.Conv2d(num_feat // 4, 1, 1, 1, 0)
        self.GELU = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        y = self.sigmoid(self.GELU(self.conv3(self.conv2(self.conv1(torch.cat((avg_pool, max_pool), dim=1))))))
        return y * x


class CALayer(nn.Module):
    def __init__(self, num_feat):
        super(CALayer, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class ResidualBlock(nn.Module):
    def __init__(self, num_feat):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.GElU = nn.GELU()
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.ca = ChannelAttention(num_feat)
        self.sa = SptialAttention(num_feat)

    def forward(self, x):
        x1 = self.GElU(self.conv1(x))
        x2 = self.GElU(self.conv2(x1))
        x3 = self.sa(self.ca(x2))
        return x + x3


class EBlock(nn.Module):
    def __init__(self, num_in_ch, num_out_ch, kernel_size, stride, padding):
        super(EBlock, self).__init__()

        self.conv = nn.Conv2d(num_in_ch, num_out_ch, kernel_size, stride, padding)
        self.GElU = nn.GELU()
        self.resb0 = ResidualBlock(num_out_ch)
        self.resb1 = ResidualBlock(num_out_ch)

    def forward(self, x):
        x1 = self.GElU(self.conv(x))
        x2 = self.resb0(x1)
        x3 = self.resb1(x2)
        return x3


class DBlock(nn.Module):
    def __init__(self, num_in_ch, num_out_ch, kernel_size, stride, padding, scale):
        super(DBlock, self).__init__()

        self.scale = scale

        self.conv = nn.Conv2d(num_in_ch, num_out_ch, kernel_size, stride, padding)
        self.GElU = nn.GELU()
        self.resb0 = ResidualBlock(num_out_ch)
        self.resb1 = ResidualBlock(num_out_ch)

    def forward(self, x):
        if self.scale == 2:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        x1 = self.GElU(self.conv(x))
        x2 = self.resb0(x1)
        x3 = self.resb1(x2)
        return x3


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale):
        super(Upsample, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels * scale * scale, 3, 1, 1)
        self.pixelshuffle = nn.PixelShuffle(upscale_factor=scale)

    def forward(self, x):
        return self.pixelshuffle(self.conv(x))


class SUNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=[32, 64, 96, 128], scale=1):
        super(SUNet, self).__init__()

        self.scale = scale

        self.conv_first = nn.Sequential(nn.Conv2d(num_in_ch, num_feat[0], 3, 1, 1),
                                        nn.GELU())

        self.eb0 = EBlock(num_feat[0], num_feat[0], 3, 1, 1)
        self.eb1 = EBlock(num_feat[0], num_feat[1], 3, 2, 1)
        self.eb2 = EBlock(num_feat[1], num_feat[2], 3, 2, 1)
        self.eb3 = EBlock(num_feat[2], num_feat[3], 3, 2, 1)

        self.db0 = DBlock(num_feat[3], num_feat[2], 3, 1, 1, 2)
        self.db1 = DBlock(num_feat[2], num_feat[1], 3, 1, 1, 2)
        self.db2 = DBlock(num_feat[1], num_feat[0], 3, 1, 1, 2)
        self.db3 = DBlock(num_feat[0], num_feat[0], 3, 1, 1, 1)

        self.conv_after_body = nn.Sequential(nn.Conv2d(num_feat[0], num_feat[0], 3, 1, 1),
                                             nn.GELU())
        self.upsample = Upsample(num_feat[0], num_out_ch, scale)
        self.conv_nr = nn.Conv2d(num_feat[0], num_out_ch, 3, 1, 1)

    def get_model_name(self):

        return "SUNet_scale" + str(self.scale) + "x"

    def forward(self, x):

        if self.scale > 1:
            x_up = F.interpolate(x, scale_factor=self.scale, mode="nearest")

        x1 = self.conv_first(x)
        x2 = self.eb0(x1)
        x3 = self.eb1(x2)
        x4 = self.eb2(x3)
        x5 = self.eb3(x4)

        out = self.db0(x5)
        out = self.db1(x4 + out)
        out = self.db2(x3 + out)

        out = self.db3(x2 + out)
        if self.scale == 1:
            y = self.conv_after_body(out)
            y = self.conv_nr(y) + x
        else:
            y = self.conv_after_body(out)
            y = self.upsample(y) + x_up
        return y


'''
net = SUNet(scale=1)

x = torch.randn((1, 3, 512, 512))

dynamic_axes = {'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                'output': {0: 'batch_size', 2: 'height', 3: 'width'}}

torch.onnx.export(net, x, 'sunet1x.onnx', export_params=True, opset_version=16, input_names=['input'],
                  output_names=['output'], dynamic_axes=dynamic_axes)
'''
