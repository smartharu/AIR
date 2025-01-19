import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.model import Model
import nets.basicblock as B
from typing import Sequence


class UNetResA(Model):
    def __init__(self, in_nc: int = 3, out_nc: int = 3, nc: Sequence[int] | None = None,
                 nb: Sequence[int] | None = None, bias=False,upsample:str = "pixelshuffle",downsample:str="strideconv"):
        super(UNetResA, self).__init__()

        if nc is None:
            nc = [48, 96, 144, 192]
        if nb is None:
            nb = [4, 2, 2, 4]

        if len(nc) != 4 or len(nb) != 4:
            raise ValueError("list len error")

        self.upsample = upsample
        self.downsample = downsample

        self.nb = nb

        self.reduction = 1

        self.head = nn.Conv2d(in_channels=in_nc, out_channels=nc[0], kernel_size=3, stride=1, padding=1, bias=False)

        self.down1 = B.sequential(
            B.ResidualChannelAttentionGroup(nc[0], 3, 1, 1, bias, self.reduction, nb[0]),
            B.DownSample(2, self.downsample, 3, 1, nc[0], nc[1], False))

        self.down2 = B.sequential(
            B.ResidualChannelAttentionGroup(nc[1], 3, 1, 1, bias, self.reduction, nb[1]),
            B.DownSample(2, self.downsample, 3, 1, nc[1], nc[2], False))
        self.down3 = B.sequential(
            B.ResidualChannelAttentionGroup(nc[2], 3, 1, 1, bias, self.reduction, nb[2]),
            B.DownSample(2, self.downsample, 3, 1, nc[2], nc[3], False))

        self.body = B.sequential(B.ResidualChannelAttentionGroup(nc[3], 3, 1, 1, bias, self.reduction, nb[3]))

        self.up3 = B.sequential(B.UpSample(2, self.upsample, 3, 1, nc[3], nc[2], False),
                                B.ResidualChannelAttentionGroup(nc[2], 3, 1, 1, bias, self.reduction, nb[2]))

        self.up2 = B.sequential(B.UpSample(2, self.upsample, 3, 1, nc[2], nc[1], False),
                                B.ResidualChannelAttentionGroup(nc[1], 3, 1, 1, bias, self.reduction, nb[1]))

        self.up1 = B.sequential(B.UpSample(2, self.upsample, 3, 1, nc[1], nc[0], False),
                                B.ResidualChannelAttentionGroup(nc[0], 3, 1, 1, bias, self.reduction, nb[0]))

        self.tail = nn.Conv2d(in_channels=nc[0], out_channels=out_nc, kernel_size=3, stride=1, padding=1, bias=False)

    def get_model_name(self):
        return "UNetResA"

    def forward(self, x0):
        x1 = self.head(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.body(x4)

        x = self.up3(x + x4)
        x = self.up2(x + x3)
        x = self.up1(x + x2)
        x = self.tail(x + x1)
        return x

class UpUNetResA(Model):
    def __init__(self, in_nc: int = 3, out_nc: int = 3, scale:int = 2,nc: Sequence[int] | None = None,
                 nb: Sequence[int] | None = None, bias=False,upsample:str = "pixelshuffle",downsample:str="strideconv"):
        super(UpUNetResA, self).__init__()

        if nc is None:
            nc = [48, 96, 144, 192]
        if nb is None:
            nb = [4, 2, 2, 4]

        if len(nc) != 4 or len(nb) != 4:
            raise ValueError("list len error")

        self.upsample = upsample
        self.downsample = downsample

        self.nb = nb

        self.reduction = 1

        self.scale = scale

        self.head = nn.Conv2d(in_channels=in_nc, out_channels=nc[0], kernel_size=3, stride=1, padding=1, bias=False)

        self.down1 = B.sequential(
            B.ResidualChannelAttentionGroup(nc[0], 3, 1, 1, bias, self.reduction, nb[0]),
            B.DownSample(2, self.downsample, 3, 1, nc[0], nc[1], False))

        self.down2 = B.sequential(
            B.ResidualChannelAttentionGroup(nc[1], 3, 1, 1, bias, self.reduction, nb[1]),
            B.DownSample(2, self.downsample, 3, 1, nc[1], nc[2], False))
        self.down3 = B.sequential(
            B.ResidualChannelAttentionGroup(nc[2], 3, 1, 1, bias, self.reduction, nb[2]),
            B.DownSample(2, self.downsample, 3, 1, nc[2], nc[3], False))

        self.body = B.sequential(B.ResidualChannelAttentionGroup(nc[3], 3, 1, 1, bias, self.reduction, nb[3]))

        self.up3 = B.sequential(B.UpSample(2, self.upsample, 3, 1, nc[3], nc[2], False),
                                B.ResidualChannelAttentionGroup(nc[2], 3, 1, 1, bias, self.reduction, nb[2]))

        self.up2 = B.sequential(B.UpSample(2, self.upsample, 3, 1, nc[2], nc[1], False),
                                B.ResidualChannelAttentionGroup(nc[1], 3, 1, 1, bias, self.reduction, nb[1]))

        self.up1 = B.sequential(B.UpSample(2, self.upsample, 3, 1, nc[1], nc[0], False),
                                B.ResidualChannelAttentionGroup(nc[0], 3, 1, 1, bias, self.reduction, nb[0]))

        self.pixel_shuffle = B.UpSamplePixelShuffle(scale=self.scale,in_channels=nc[0],out_channels=nc[0])

        self.tail = nn.Conv2d(in_channels=nc[0], out_channels=out_nc, kernel_size=3, stride=1, padding=1, bias=False)

    def get_model_name(self):
        return "UpUNetResA"

    def forward(self, x0):
        x1 = self.head(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.body(x4)

        x = self.up3(x + x4)
        x = self.up2(x + x3)
        x = self.up1(x + x2)
        x = self.pixel_shuffle(x + x1)
        x = self.tail(x)
        return x

if __name__ == "__main__":
    x = torch.randn((1, 3, 128, 128))


    net = UNetResA(3, 3, [32, 64, 128, 256], [4, 4, 4, 4], True)
    #net = UpUNetResA(3, 3, 2, [32, 64, 128, 256], [4, 4, 4, 4], True)
    #net = UNetResA(3, 3, [48, 96, 144, 192], [4, 2, 2, 4], True)

    # net = SPAU()
    # net = UNetResAMOD()
    # net = SCUNetRes()

    # net = UNetResAMOD(3, 3, [32, 64, 96, 128], 4, True)

    # net = UNetResASubPixel(3, 3, [64, 128, 256], 4, True)

    # net = UNetResASubP(3, 3, [32, 64, 96, 128], 4, True, 1)

    x = net(x)

    print(x.shape)

    net.load_model()

    net.convert_to_onnx()

    # net = NonLocalUNetResA()

    # net.convert_to_onnx()
