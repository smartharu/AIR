import sys
import torch
import torchvision
import torchvision.transforms.functional as TTF
import torchvision.transforms as TT
import random
import math

def get_platform() -> str:
    if sys.platform.startswith("linux"):
        return "linux"

    elif sys.platform.startswith("win32"):
        return "win"
    else:
        raise ValueError("current platform not supported")

def limit_number(num:int|float, min:int|float, max:int|float) -> int|float:
    if num < min:
        num = min
    if num > max:
        num = max
    return num

def crop(img: torch.Tensor, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0) -> torch.Tensor:
    c, h, w = img.shape
    ret = TTF.crop(img, left=left, top=top, height=h - top - bottom,width=w - left - right)
    return ret

def add_border(img: torch.Tensor, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0) -> torch.Tensor:
    ret = TTF.pad(img, padding=[left, top, right, bottom], padding_mode="constant")
    return ret

def load_image(img:str) -> torch.Tensor:
    mode = torchvision.io.image.ImageReadMode.RGB
    image = torchvision.io.read_image(img, mode)
    return image.float() / 255.

class RandomSafeRotate:
    def __init__(self, prob: float = 0.3) -> None:
        self.prob = prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.uniform(0, 1) > self.prob:
            return x
        c, h, w = x.shape
        p = math.ceil(math.sqrt(2) / 4 * max(h, w))
        pad = TTF.pad(x, padding=[p, p, p, p], padding_mode="reflect")

        ang = random.randint(-45, 45)
        rot = TTF.rotate(pad, angle=ang,interpolation=TT.InterpolationMode.BILINEAR)
        ret = TTF.center_crop(rot, output_size=[h, w])
        return ret

class RandomRotate:
    def __init__(self, prob: float = 0.5) -> None:
        self.prob = prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.uniform(0, 1) > self.prob:
            return x
        c, h, w = x.shape

        ang = random.choice([-90, 90])
        rot = TTF.rotate(x, angle=ang,interpolation=TT.InterpolationMode.BILINEAR)
        return rot

class RamdomAugBorder:
    def __init__(self, prob: float = 0.5) -> None:
        self.prob = prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.uniform(0, 1) > self.prob:
            return x
        c, h, w = x.shape

        left = random.randint(0, 8)
        right = random.randint(0, 8)
        top = random.randint(0, 8)
        bottom = random.randint(0, 8)

        c = crop(x, left, right, top, bottom)
        a = add_border(c, left, right, top, bottom)
        return a