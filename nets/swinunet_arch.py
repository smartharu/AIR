import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.swin_transformer import SwinTransformerBlock
from nets.model import Model
import nets.basicblock as B


def NO_NORM_LAYER(dim):
    return nn.Identity()


class SwinTransformerBlocks(nn.Module):
    def __init__(self, in_channels: int, num_head: int, num_layers: int, window_size: list[int], norm_layer: nn.Module):
        super().__init__()
        layers = []
        for i_layer in range(num_layers):
            layers.append(
                SwinTransformerBlock(
                    in_channels,
                    num_head,
                    window_size=window_size,
                    shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                    mlp_ratio=2.,
                    dropout=0.,
                    attention_dropout=0.,
                    stochastic_depth_prob=0.,
                    norm_layer=norm_layer,
                )
            )
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        y = self.block(x)
        return y


class PatchDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)

    def forward(self, x: torch.Tensor):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()  # BHWC->BCHW
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # BCHW->BHWC
        return x


class PatchUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 3, 1, 2)  # BHWC->BCHW
        x = self.conv(x)
        x = F.pixel_shuffle(x, 2)
        x = x.permute(0, 2, 3, 1).contiguous()  # BCHW->BHWC
        return x


class SwinUNet(Model):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_dim: int = 32, base_layers: int = 2,
                 norm_layer: nn.Module = NO_NORM_LAYER):
        super().__init__()
        num_heads: int = base_dim // 16
        window_size: list[int] = [8, 8]

        self.head = nn.Conv2d(in_channels, base_dim, kernel_size=3, stride=1, padding=1)

        self.down1 = nn.Sequential(
            SwinTransformerBlocks(base_dim, num_head=num_heads, num_layers=base_layers, window_size=window_size,
                                  norm_layer=norm_layer),
            PatchDown(base_dim, base_dim * 2))

        self.down2 = nn.Sequential(
            SwinTransformerBlocks(base_dim * 2, num_head=num_heads, num_layers=base_layers, window_size=window_size,
                                  norm_layer=norm_layer),
            PatchDown(base_dim * 2, base_dim * 4))

        self.down3 = nn.Sequential(
            SwinTransformerBlocks(base_dim * 4, num_head=num_heads, num_layers=base_layers, window_size=window_size,
                                  norm_layer=norm_layer),
            PatchDown(base_dim * 4, base_dim * 8))

        self.body = SwinTransformerBlocks(base_dim * 8, num_head=num_heads, num_layers=base_layers,
                                          window_size=window_size, norm_layer=norm_layer)

        self.up3 = nn.Sequential(PatchUp(base_dim * 8, base_dim * 4),
                                 SwinTransformerBlocks(base_dim * 4, num_head=num_heads, num_layers=base_layers,
                                                       window_size=window_size, norm_layer=norm_layer))

        self.up2 = nn.Sequential(PatchUp(base_dim * 4, base_dim * 2),
                                 SwinTransformerBlocks(base_dim * 2, num_head=num_heads, num_layers=base_layers,
                                                       window_size=window_size, norm_layer=norm_layer))

        self.up1 = nn.Sequential(PatchUp(base_dim * 2, base_dim),
                                 SwinTransformerBlocks(base_dim, num_head=num_heads, num_layers=base_layers,
                                                       window_size=window_size, norm_layer=norm_layer))

        self.tail = nn.Conv2d(base_dim, out_channels, kernel_size=3, stride=1, padding=1)

    def get_model_name(self):
        return "SwinUNet"

    def forward(self, x: torch.Tensor):
        x1 = self.head(x)
        x2 = x1.permute(0, 2, 3, 1).contiguous()  # BHWC

        x3 = self.down1(x2)
        x4 = self.down2(x3)
        x5 = self.down3(x4)

        x = self.body(x5)

        x = self.up3(x + x5)
        x = self.up2(x + x4)
        x = self.up1(x + x3)

        x = x.permute(0, 3, 1, 2).contiguous()  # BCHW
        x = self.tail(x + x1)
        return x


class SwinConvUNet(Model):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_dim: int = 32, num_blocks: int = 2,
                 norm_layer: nn.Module = NO_NORM_LAYER, upsample: str = "pixelshuffle", downsample: str = "strideconv"):
        super().__init__()
        num_heads: int = base_dim // 16
        window_size: list[int] = [8, 8]

        self.head = nn.Conv2d(in_channels, base_dim, kernel_size=3, stride=1, padding=1)

        self.down1 = nn.Sequential(
            B.ConvTransGroup(base_dim, base_dim, num_heads, window_size, num_blocks, norm_layer),
            B.DownSample(2, downsample, 2, 1, base_dim, base_dim * 2, False))

        self.down2 = nn.Sequential(
            B.ConvTransGroup(base_dim * 2, base_dim * 2, num_heads, window_size, num_blocks, norm_layer),
            B.DownSample(2, downsample, 2, 1, base_dim * 2, base_dim * 4, False))

        self.down3 = nn.Sequential(
            B.ConvTransGroup(base_dim * 4, base_dim * 4, num_heads, window_size, num_blocks, norm_layer),
            B.DownSample(2, downsample, 2, 1, base_dim * 4, base_dim * 8, False))

        self.body = B.ConvTransGroup(base_dim * 8, base_dim * 8, num_heads, window_size, num_blocks, norm_layer)

        self.up3 = nn.Sequential(B.UpSample(2, upsample, 3, 1, base_dim * 8, base_dim * 4, False),
                                 B.ConvTransGroup(base_dim * 4, base_dim * 4, num_heads, window_size, num_blocks,
                                                  norm_layer))

        self.up2 = nn.Sequential(B.UpSample(2, upsample, 3, 1, base_dim * 4, base_dim * 2, False),
                                 B.ConvTransGroup(base_dim * 2, base_dim * 2, num_heads, window_size, num_blocks,
                                                  norm_layer))

        self.up1 = nn.Sequential(B.UpSample(2, upsample, 3, 1, base_dim * 2, base_dim, False),
                                 B.ConvTransGroup(base_dim, base_dim, num_heads, window_size, num_blocks, norm_layer))

        self.tail = nn.Conv2d(base_dim, out_channels, kernel_size=3, stride=1, padding=1)

    def get_model_name(self):
        return "SwinConvUNet"

    def forward(self, x: torch.Tensor):
        x1 = self.head(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.body(x4)

        x = self.up3(x + x4)
        x = self.up2(x + x3)
        x = self.up1(x + x2)
        x = self.tail(x + x1)
        return x


if __name__ == "__main__":
    # net = SwinUNet(in_channels=3, out_channels=3, base_dim=32, base_layers=4, norm_layer=NO_NORM_LAYER)
    # net.load_model()
    # net.to_inference_model()
    # net.convert_to_onnx()

    net = SwinConvUNet(3, 3, 32, 2)
    net.convert_to_onnx()
