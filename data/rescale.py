import torch
import torchvision.transforms as TT
import torchvision.transforms.functional as TTF
import torch.nn.functional as TF
import random
from data.transform import resize
from data.utils import limit_number

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
            "half_length": 64,
            "third_length": 42,
        },
        144: {
            "min_length": 54,
            "max_length": 144,
            "anisotropic_length": 36,
            "half_length": 72,
            "third_length": 48,
        },
        64: {
            "min_length": 24,
            "max_length": 64,
            "anisotropic_length": 16,
            "half_length": 32,
            "third_length": 20,
        },
        80: {
            "min_length": 30,
            "max_length": 80,
            "anisotropic_length": 20,
            "half_length": 40,
            "third_length": 24,
        }
    },
    "second": {
        128: {
            "min_length": 96,
            "max_length": 128,
            "anisotropic_length": 32,
            "half_length": 64,
            "third_length": 42,
        },
        144: {
            "min_length": 108,
            "max_length": 144,
            "anisotropic_length": 36,
            "half_length": 72,
            "third_length": 48,
        },
        64: {
            "min_length": 48,
            "max_length": 64,
            "anisotropic_length": 16,
            "half_length": 32,
            "third_length": 20,
        },
        80: {
            "min_length": 60,
            "max_length": 80,
            "anisotropic_length": 20,
            "half_length": 40,
            "third_length": 24,
        }
    },
    "aa": {
        128: {
            "min_length": 128,
            "max_length": 320,
            "anisotropic_length": 128,
            "double_length": 256,
        },
        144: {
            "min_length": 144,
            "max_length": 360,
            "anisotropic_length": 144,
            "double_length": 288,
        },
        64: {
            "min_length": 64,
            "max_length": 160,
            "anisotropic_length": 64,
            "double_length": 20,
        },
        80: {
            "min_length": 80,
            "max_length": 200,
            "anisotropic_length": 80,
            "double_length": 160,
        }
    },
    "nr": {
        128: {
            "min_length": 48,
            "max_length": 64,
            "anisotropic_length": 32,
            "half_length": 64,
            "third_length": 42,
        },
        144: {
            "min_length": 54,
            "max_length": 72,
            "anisotropic_length": 36,
            "half_length": 72,
            "third_length": 48,
        },
        64: {
            "min_length": 24,
            "max_length": 32,
            "anisotropic_length": 16,
            "half_length": 32,
            "third_length": 20,
        },
        80: {
            "min_length": 30,
            "max_length": 40,
            "anisotropic_length": 20,
            "half_length": 40,
            "third_length": 24,
        }
    },
}


def rescale(x: torch.Tensor, task: str | None=None, min_length: int | None=None, max_length: int | None=None,
            anisotropic_length: int | None=None, half_length: int | None=None, third_length: int | None=None,
            double_length: int | None=None, anisotropic_p: float = 0.3) -> torch.Tensor:
    if len(x.shape) <= 2:
        x = x.unsqueeze(0)
        gray = True
    else:
        gray = False

    source_width = x.shape[2]
    source_height = x.shape[1]

    kernel_type = [KERNEL_BOX, KERNEL_BILINEAR, KERNEL_BICUBIC, KERNEL_WINDOW]
    kernel_type_wo_box = [KERNEL_BILINEAR, KERNEL_BICUBIC, KERNEL_WINDOW]

    kernel_type_weight = [1, 1 / 2, 1, 1]
    kernel_type_weight_wo_box = [1 / 2, 1, 1]

    kernel_type_choice = random.choices(kernel_type, weights=kernel_type_weight, k=2)
    kernel_type_choice_wo_box = random.choices(kernel_type_wo_box, weights=kernel_type_weight_wo_box, k=2)

    interpolation_kernels_up = random.choice(kernel_type_choice[0])
    interpolation_kernels_down = random.choice(kernel_type_choice[1])
    interpolation_kernels_pre_box = random.choice(kernel_type_choice_wo_box[0])
    interpolation_kernels_back = random.choice(kernel_type_choice_wo_box[1])

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

    #print("target_width = ",target_width," target_height = ",target_height)

    if task == "first" or task == "second" or task == "nr":

        if interpolation_kernels_down == "box":
            if (target_height == half_length and target_width == half_length) or (
                    target_height == third_length and target_width == third_length):
                x = resize(x, target_width, target_height, "box")
            elif min(target_width, target_height) > half_length:
                x = resize(x, target_width * 2, target_height * 2, interpolation_kernels_pre_box)
                x = resize(x, target_width, target_height, "box")
            else:
                x = resize(x, target_width * 3, target_height * 3, interpolation_kernels_pre_box)
                x = resize(x, target_width, target_height, "box")
        else:
            x = resize(x, target_width, target_height, interpolation_kernels_down)

        if interpolation_kernels_up == "box":
            if (target_height == half_length and target_width == half_length) or (
                    target_height == third_length and target_width == third_length):
                x = resize(x, source_width, source_height, "box")
            elif min(target_width, target_height) > half_length:
                x = resize(x, target_width * 2, target_height * 2, "box")
                x = resize(x, source_width, source_height, interpolation_kernels_back)
            else:
                x = resize(x, target_width * 3, target_height * 3, "box")
                x = resize(x, source_width, source_height, interpolation_kernels_back)
        else:
            x = resize(x, source_width, source_height, interpolation_kernels_up)

    elif task == "aa":

        if interpolation_kernels_up == "box":
            if target_height == double_length and target_width == double_length:
                x = resize(x, target_width, target_height, "box")
            elif max(target_height, target_width) < double_length:
                x = resize(x, source_width * 2, source_height * 2, "box")
                x = resize(x, target_width, target_height, interpolation_kernels_pre_box)
            else:
                x = resize(x, source_width * 3, source_height * 3, "box")
                x = resize(x, target_width, target_height, interpolation_kernels_pre_box)
        else:
            x = resize(x, target_width, target_height, interpolation_kernels_up)

        if interpolation_kernels_down == "box":
            if target_height == double_length and target_width == double_length:
                x = resize(x, source_width, source_width, "box")
            elif max(target_height, target_width) < double_length:
                x = resize(x, source_width * 2, source_height * 2, interpolation_kernels_back)
                x = resize(x, source_width, source_height, "box")
            else:
                x = resize(x, source_width * 3, source_height * 3, interpolation_kernels_back)
                x = resize(x, source_width, source_height, "box")
        else:
            x = resize(x, source_width, source_height, interpolation_kernels_down)

    else:
        raise ValueError("task must be first second nr or aa")

    if gray:
        x = x.squeeze(0)
    return x


class Waifu2xRandomDownscale:
    def __init__(self, scale_factor: int = 2) -> None:
        self.scale_factor = scale_factor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        source_width = x.shape[2]
        source_height = x.shape[1]

        kernel_type = [KERNEL_BOX, KERNEL_BILINEAR, KERNEL_BICUBIC, KERNEL_WINDOW]

        kernel_type_weight = [1, 1 / 2, 1, 1]
        kernel_type_choice = random.choices(kernel_type, weights=kernel_type_weight, k=1)

        kernel = random.choice(kernel_type_choice[0])
        x = resize(x, source_width // self.scale_factor, source_height // self.scale_factor, kernel)

        return x

class A4KRandomDownscale:
    def __init__(self, scale_factor: int = 2) -> None:
        self.scale_factor = scale_factor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        source_width = x.shape[2]
        source_height = x.shape[1]

        kernel_type = [KERNEL_BOX, KERNEL_BILINEAR, KERNEL_BICUBIC, KERNEL_WINDOW]

        kernel_type_weight = [5/2, 1 / 2, 1, 1]
        kernel_type_choice = random.choices(kernel_type, weights=kernel_type_weight, k=1)

        kernel = random.choice(kernel_type_choice[0])
        x = resize(x, source_width // self.scale_factor, source_height // self.scale_factor, kernel)

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

        if self.task == "aa":
            min_length = RS_RATE[self.task][source_width]["min_length"]
            max_length = RS_RATE[self.task][source_width]["max_length"]
            anisotropic_length = RS_RATE[self.task][source_width]["anisotropic_length"]
            double_length = RS_RATE[self.task][source_width]["double_length"]
            rescale(x, self.task, min_length=min_length, max_length=max_length, anisotropic_length=anisotropic_length,
                    double_length=double_length, anisotropic_p=self.anisotropic_p)
        elif self.task == "first" or self.task == "second" or self.task == "nr":
            min_length = RS_RATE[self.task][source_width]["min_length"]
            max_length = RS_RATE[self.task][source_width]["max_length"]
            anisotropic_length = RS_RATE[self.task][source_width]["anisotropic_length"]
            half_length = RS_RATE[self.task][source_width]["half_length"]
            third_length = RS_RATE[self.task][source_width]["third_length"]
            #print(self.task,min_length,max_length,anisotropic_length,half_length,third_length)
            rescale(x, self.task, min_length=min_length, max_length=max_length, anisotropic_length=anisotropic_length,
                    half_length=half_length, third_length=third_length, anisotropic_p=self.anisotropic_p)
        else:
            raise ValueError("task must be first second nr or aa")

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

'''import torchvision
def load_images(dir: str) -> torch.Tensor:
    mode = torchvision.io.image.ImageReadMode.RGB
    image = torchvision.io.read_image(dir, mode)
    return image.float() / 255.

if __name__ == "__main__":
    img = load_images("SYNLA_NEO_143.png")
    img = AntialiasX(prob=1.0)(img)
    img = TT.ToPILImage()(img)
    img.save("out2.png")'''
