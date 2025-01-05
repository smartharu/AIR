import torch
import torchvision.transforms as TT
import torchvision.transforms.functional as TTF
import torch.nn.functional as TF
import random
from data.transform import resize
from data.datatools import limit_number, load_image, crop

KERNEL_BOX: list = ["box"]
KERNEL_BILINEAR: list = ["triangle"]
KERNEL_BICUBIC: list = ["hermite", "catrom", "mitchell", "robidoux", "robidouxsharp", "lagrange", "spline"]
KERNEL_WINDOW: list = ["lanczos", "lanczossharp", "lanczos2", "lanczos2sharp", "jinc", "sinc",
                       "hanning", "hamming",
                       "blackman", "kaiser", "welsh", "parzen", "bohman", "bartlett", "cosine", "sentinel"]

RS_RATE = {
    "first": {
        128: {
            "min_length": 48,
            "max_length": 128,
            "anisotropic_length": 32,
        },
        144: {
            "min_length": 54,
            "max_length": 144,
            "anisotropic_length": 36,
        },
        64: {
            "min_length": 24,
            "max_length": 64,
            "anisotropic_length": 16,
        },
        80: {
            "min_length": 30,
            "max_length": 80,
            "anisotropic_length": 20,
        }
    },
    "second": {
        128: {
            "min_length": 96,
            "max_length": 128,
            "anisotropic_length": 32,
        },
        144: {
            "min_length": 108,
            "max_length": 144,
            "anisotropic_length": 36,
        },
        64: {
            "min_length": 48,
            "max_length": 64,
            "anisotropic_length": 16,
        },
        80: {
            "min_length": 60,
            "max_length": 80,
            "anisotropic_length": 20,
        }
    },
    "aa": {
        128: {
            "min_length": 128,
            "max_length": 320,
            "anisotropic_length": 128,
        },
        144: {
            "min_length": 144,
            "max_length": 360,
            "anisotropic_length": 144,
        },
        64: {
            "min_length": 64,
            "max_length": 160,
            "anisotropic_length": 64,
        },
        80: {
            "min_length": 80,
            "max_length": 200,
            "anisotropic_length": 80,
        }
    },
    "nr": {
        128: {
            "min_length": 48,
            "max_length": 64,
            "anisotropic_length": 32,
        },
        144: {
            "min_length": 54,
            "max_length": 72,
            "anisotropic_length": 36,
        },
        64: {
            "min_length": 24,
            "max_length": 32,
            "anisotropic_length": 16,
        },
        80: {
            "min_length": 30,
            "max_length": 40,
            "anisotropic_length": 20,
        }
    },
}


def rescale(x: torch.Tensor, min_length:int = None, max_length:int=None,anisotropic_p: float = 0.3,anisotropic_length:int=None) -> torch.Tensor:
    if len(x.shape) <= 2:
        x = x.unsqueeze(0)
        gray = True
    else:
        gray = False

    source_width = x.shape[2]
    source_height = x.shape[1]

    kernel_type = [KERNEL_BOX, KERNEL_BILINEAR, KERNEL_BICUBIC, KERNEL_WINDOW]
    kernel_type_weight = [1, 1 / 2, 1, 1]

    kernel_type_choice = random.choices(kernel_type, weights=kernel_type_weight, k=2)

    interpolation_kernels_first = random.choice(kernel_type_choice[0])
    interpolation_kernels_second = random.choice(kernel_type_choice[1])

    target_width = random.randint(min_length, max_length)
    target_height = target_width

    if random.uniform(0, 1) < anisotropic_p:
        scale_shift = random.randint(0, anisotropic_length)
        if random.uniform(0, 1) < 0.5:
            target_height += scale_shift
        else:
            target_height -= scale_shift

        target_height = limit_number(target_height, min_length, max_length)

    if random.uniform(0, 1) < 0.5:
        target_width, target_height = target_height, target_width

    x = resize(x, target_width, target_height, interpolation_kernels_first)
    x = resize(x, source_width, source_height, interpolation_kernels_second)

    if gray:
        x = x.squeeze(0)
    return x

class RandomRescale:
    def __init__(self, prob: float = 0.3, task: str = "first", anisotropic_p: float = 0.3) -> None:
        self.prob = prob
        self.anisotropic_p = anisotropic_p
        self.task = task

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.uniform(0, 1) > self.prob:
            return x
        source_width = x.shape[2]
        source_height = x.shape[1]

        if self.task in ["first","second","aa","nr"]:
            x = rescale(x,RS_RATE[self.task][source_width]["min_length"],RS_RATE[self.task][source_width]["max_length"],self.anisotropic_p,RS_RATE[self.task][source_width]["anisotropic_length"])
        else:
            raise ValueError("task must be first second or aa")

        return x.clamp(0, 1)


class AntialiasX:
    """
    antialias used in Waifu2x
    """

    def __init__(self, prob: float = 0.05):
        self.prob = prob
        pass

    def __call__(self, x: torch.Tensor):
        if random.uniform(0, 1) > self.prob:
            return x
        B,W, H = x.shape
        interpolation = random.choice([TT.InterpolationMode.BICUBIC, TT.InterpolationMode.BILINEAR])
        if random.uniform(0, 1) < 0.5:
            scale = 2
        else:
            scale = random.uniform(1.5, 2)
        x = TTF.resize(x, (int(H * scale), int(W * scale)), interpolation=interpolation, antialias=True)
        x = TTF.resize(x, (H, W), interpolation=TT.InterpolationMode.BICUBIC, antialias=True)
        return x

if __name__ == "__main__":
    img = load_image("SYNLA_NEO_143.png")
    img = crop(img,64,64,64,64)
    img = RandomRescale(prob=1.0)(img)
    img = TT.ToPILImage()(img)
    img.save("out2.png")
