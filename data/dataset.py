import random
import time
import numpy
import numpy as np
import torch
import torchvision
from torch.nn import functional as F
from torch.utils import data
import os
import math
import cv2
from scipy import special
from typing import Any, Mapping, Optional, Sequence, Union, Tuple
from wand.image import Image, IMAGE_TYPES
import wand

GRAYSCALE_TYPES = {
    "grayscale",
    "grayscalematte",
    "grayscalealpha",
}
GRAYSCALE_ALPHA_TYPE = "grayscalealpha" if "grayscalealpha" in IMAGE_TYPES else "grayscalematte"
GRAYSCALE_TYPE = "grayscale"
RGBA_TYPE = "truecoloralpha" if "truecoloralpha" in IMAGE_TYPES else "truecolormatte"
RGB_TYPE = "truecolor"
GAMMA_LCD = 45454


def read_images(dir: str, is_train: bool = True) -> tuple[list[torch.Tensor], list[str]]:
    mode = torchvision.io.image.ImageReadMode.RGB
    labels = []
    filenames = []
    if is_train:
        train_list = os.listdir(os.path.join(dir, "Dataset", "train"))
        for train_img in train_list:
            if train_img.split(".").pop() in ["png"]:
                img = torchvision.io.read_image(os.path.join(dir, "Dataset", "train", train_img), mode)
                labels.append(img)
                filenames.append(train_img)
    else:
        valid_list = os.listdir(os.path.join(dir, "Dataset", "valid"))
        for valid_img in valid_list:
            if valid_img.split(".").pop() in ["png"]:
                img = torchvision.io.read_image(os.path.join(dir, "Dataset", "valid", valid_img), mode)
                labels.append(img)
                filenames.append(valid_img)
    return labels, filenames


def predict_storage(dtype: torch.dtype, int_type: str = "short") -> str:
    if dtype in {torch.float, torch.float32, torch.float16}:
        storage = "float"
    elif dtype in {torch.double, torch.float64}:
        storage = "double"
    elif dtype == torch.uint8:
        storage = "char"
    else:
        storage = int_type
    return storage


def to_wand_image(img: torch.Tensor) -> wand.image:
    ch, h, w = img.shape
    assert (ch in {1, 3})
    if ch == 1:
        channel_map = "I"
    else:
        channel_map = "RGB"

    storage = predict_storage(img.dtype, int_type="long")

    arr = img.permute(1, 2, 0).detach().cpu().numpy()
    return Image.from_array(arr, channel_map=channel_map, storage=storage)


def to_tensor(img: wand.image, dtype=torch.float32) -> torch.Tensor:
    if img.type in {RGB_TYPE, RGBA_TYPE}:
        channel_map = "RGB"
    elif img.type in {GRAYSCALE_TYPE, GRAYSCALE_ALPHA_TYPE}:
        channel_map = "R"
    else:
        assert (img.type in {RGB_TYPE, RGBA_TYPE, GRAYSCALE_TYPE, GRAYSCALE_ALPHA_TYPE})

    storage = predict_storage(dtype)
    w, h = img.size
    ch = len(channel_map)
    data = img.export_pixels(0, 0, w, h, channel_map=channel_map, storage=storage)
    x = torch.tensor(data, dtype=dtype).view(h, w, ch).permute(2, 0, 1).contiguous()
    del data
    return x


def resize(img: torch.Tensor, w: int, h: int, kernel: str = "cubic") -> torch.Tensor:
    with to_wand_image(img) as im:
        im.resize(width=w, height=h, filter=kernel)
        return to_tensor(im)


def jpeg_compression(img: torch.Tensor, quality: int = 90, subsampling: bool = True) -> torch.Tensor:
    img = img.permute(1, 2, 0).numpy()
    if subsampling:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    else:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality, int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR),
                        cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]
    _, encimg = cv2.imencode('.jpg', img * 255., encode_param)
    img = np.float32(cv2.imdecode(encimg, 1)) / 255.
    img = torch.from_numpy(img).permute(2, 0, 1).clamp(0, 1)
    return img


def rgb_to_yuv(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to YUV.

    .. image:: _static/img/rgb_to_yuv.png

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: RGB Image to be converted to YUV with shape :math:`(*, 3, H, W)`.

    Returns:
        YUV version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        input = torch.rand(2, 3, 4, 5)
        output = rgb_to_yuv(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    y: torch.Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    u: torch.Tensor = -0.147 * r - 0.289 * g + 0.436 * b
    v: torch.Tensor = 0.615 * r - 0.515 * g - 0.100 * b

    out: torch.Tensor = torch.stack([y, u, v], -3)

    return out


def yuv_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an YUV image to RGB.

    The image data is assumed to be in the range of (0, 1) for luma and (-0.5, 0.5) for chroma.

    Args:
        image: YUV Image to be converted to RGB with shape :math:`(*, 3, H, W)`.

    Returns:
        RGB version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        input = torch.rand(2, 3, 4, 5)
        output = yuv_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    y: torch.Tensor = image[..., 0, :, :]
    u: torch.Tensor = image[..., 1, :, :]
    v: torch.Tensor = image[..., 2, :, :]

    r: torch.Tensor = y + 1.14 * v  # coefficient for g is 0
    g: torch.Tensor = y + -0.396 * u - 0.581 * v
    b: torch.Tensor = y + 2.029 * u  # coefficient for b is 0

    out: torch.Tensor = torch.stack([r, g, b], -3)

    return out


def gaussian_noise(img: torch.Tensor, sigma: float, gray_noise: bool = False) -> torch.Tensor:
    img_yuv = rgb_to_yuv(img)
    y = img_yuv[..., 0, :, :]
    u = img_yuv[..., 1, :, :]
    v = img_yuv[..., 2, :, :]
    if gray_noise:
        imshape = y.shape
    else:
        imshape = img.shape

    rg = torch.normal(mean=0, std=sigma, size=imshape)

    if gray_noise:
        y = y + rg
        y = y.clamp(0, 1)
        img_merge = torch.stack([y, u, v], -3)
        ret = yuv_to_rgb(img_merge)
    else:
        ret = img + rg

    return ret.clamp(0, 1)


'''
def poisson_noise(img: torch.Tensor, scale: float, gray_noise: bool = False):
    img_yuv = rgb_to_yuv(img)
    y = img_yuv[..., 0, :, :]
    u = img_yuv[..., 1, :, :]
    v = img_yuv[..., 2, :, :]
    im = img.numpy()
    vals = len(np.unique(im))
    vals = 2 ** np.ceil(np.log2(vals))
    out = np.float32(np.random.poisson(im * vals) / float(vals))
    noise = out - im
    noise = noise * scale

    out = im + noise
    out = np.clip(out, 0, 1)

    if gray_noise:
        add = rgb_to_yuv(torch.from_numpy(out))
        y_add = add[..., 0, :, :]
        u_add = add[..., 1, :, :]
        v_add = add[..., 2, :, :]
        res = yuv_to_rgb(torch.stack([y_add, u, v], -3))
    else:
        res = torch.from_numpy(out)

    return res.clamp(0, 1)

'''


def box_blur(img: torch.Tensor, radius: int = 1) -> torch.Tensor:
    ksize = 2 * radius + 1

    k = torch.ones((ksize, ksize), dtype=torch.float32)
    k = k / k.sum()

    img = img.unsqueeze(0)
    B, C, H, W = img.shape
    img = F.pad(img, (radius, radius, radius, radius), mode="reflect")
    k = k.view(1, 1, ksize, ksize).repeat(C, 1, 1, 1)
    out = F.conv2d(img, weight=k, bias=None, stride=1, groups=C)
    return out.squeeze(0).clamp(0, 1)


def unsharp_mask(img: torch.Tensor, kernel_size: int = 3, sigma: float = 2.0, gray: bool = True) -> torch.Tensor:
    nr = gaussian_blur(img, kernel_size, sigma)

    makediff = (img - nr).clamp(0, 1)
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

    img = img.unsqueeze(0)
    B, C, H, W = img.shape
    img = F.pad(img, (ksize // 2, ksize // 2, ksize // 2, ksize // 2), mode="reflect")
    k = h.view(1, 1, ksize, ksize).repeat(C, 1, 1, 1)

    out = F.conv2d(img, weight=k, bias=None, stride=1, groups=C)
    return out.squeeze(0).clamp(0, 1)


def ring_filter(img: torch.Tensor, ksize: int, sigma: float) -> torch.Tensor:
    n = (ksize - 1.0) / 2.0
    x = torch.arange(-n, n + 1, dtype=torch.float32).reshape(ksize, 1)
    y = torch.arange(-n, n + 1, dtype=torch.float32).reshape(1, ksize)

    d = torch.sqrt((x * x) + (y * y))
    h = torch.sinc(d) * torch.sinc(d / sigma)
    h = torch.div(h, h.sum())

    img = img.unsqueeze(0)
    B, C, H, W = img.shape
    img = F.pad(img, (ksize // 2, ksize // 2, ksize // 2, ksize // 2), mode="reflect")
    k = h.view(1, 1, ksize, ksize).repeat(C, 1, 1, 1)

    out = F.conv2d(img, weight=k, bias=None, stride=1, groups=C)
    return out.squeeze(0).clamp(0, 1)


def random_select_kernel() -> str:
    kernel = ["box", "hermite", "triangle", "cubic", "catrom", "mitchell", "sinc", "lanczos"]
    interpolation_kernel = random.choice(kernel)
    return interpolation_kernel


class RandomJPEGNoise:
    def __init__(self, prob: float = 0.5, jpeg_q: Union[int, Tuple[int, int]] = 95,
                 css_prob: float = 0.5) -> None:
        self.prob = prob
        self.css_prob = css_prob
        self.jpeg_q = jpeg_q

    def __call__(self, x) -> torch.Tensor:
        if random.uniform(0, 1) > self.prob:
            return x

        if isinstance(self.jpeg_q, int):
            jpeg_qualities = self.jpeg_q
        elif isinstance(self.jpeg_q, tuple):
            jpeg_qualities = random.randint(self.jpeg_q[0], self.jpeg_q[1])
        else:
            raise TypeError("jepg_q 应该为int或(int,int)")

        if random.uniform(0, 1) < self.css_prob:
            chroma_subsampling = False
        else:
            chroma_subsampling = True

        x = jpeg_compression(x, jpeg_qualities, chroma_subsampling)

        return x


class RandomDownscale:
    def __init__(self, scale_factor: int = 2) -> None:
        self.scale_factor = scale_factor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        sw = x.shape[2]
        sh = x.shape[1]
        kernel = random_select_kernel()
        x = resize(x, sw // self.scale_factor, sh // self.scale_factor, kernel)

        return x


class RandomRescale:
    def __init__(self, prob: float = 0.3, scale: Tuple[float, float] = (1.0, 2.0)) -> None:
        self.prob = prob
        self.scale_range = scale

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.uniform(0, 1) > self.prob:
            return x
        sw = x.shape[2]
        sh = x.shape[1]

        kernel_down = random_select_kernel()
        kernel_up = random_select_kernel()

        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        if kernel_up == "box" or kernel_down == "box":
            scale = 2.0

        dw = math.floor(sw / scale + 0.5)
        dh = math.floor(sh / scale + 0.5)
        x = resize(x, dw, dh, kernel_down)
        x = resize(x, sw, sh, kernel_up)

        return x


class RandomGray:
    def __init__(self, prob: float = 0.3) -> None:
        self.prob = prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.uniform(0, 1) > self.prob:
            return x
        return torchvision.transforms.functional.rgb_to_grayscale(x, 3)


class RandomUSM:
    def __init__(self, prob: float = 0.3, gray_prob: float = 0.5) -> None:
        self.prob = prob
        self.gray_prob = gray_prob

        self.kernel_sizes = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
        self.kernel_weights = [1 / 3, 1 / 3, 1 / 6, 1 / 6, 1 / 6, 1 / 12, 1 / 12, 1 / 12, 1 / 12, 1 / 24]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.uniform(0, 1) > self.prob:
            return x

        sigma = random.uniform(0.1, 2.0)

        ksize = random.choices(self.kernel_sizes, weights=self.kernel_weights, k=1)[0]

        if random.uniform(0, 1) > self.gray_prob:
            gray = True
        else:
            gray = False

        sharp = unsharp_mask(x, ksize, sigma, gray)

        return sharp


class RandomNoise:
    def __init__(self, prob: float = 0.65, gaussian_factor: int = 25, poisson_factor: int = 3,
                 gray_prob: float = 0.5, gaussian_prob: float = 0.7) -> None:
        self.prob = prob
        self.gaussian_factor = gaussian_factor
        self.poisson_factor = poisson_factor
        self.gray_prob = gray_prob
        self.gaussian_prob = gaussian_prob

    def __call__(self, x) -> torch.Tensor:
        if random.uniform(0, 1) > self.prob:
            return x

        if random.uniform(0, 1) < self.gray_prob:
            gray_noise = True
        else:
            gray_noise = False

        sigma = random.randint(0, self.gaussian_factor)
        scale = random.uniform(0, self.poisson_factor)

        if random.uniform(0, 1) < self.gaussian_prob:
            x = gaussian_noise(x, sigma / 255., gray_noise)
        else:
            # x = poisson_noise(x, self.scale, self.gray_noise)
            x = gaussian_noise(x, sigma / 255., gray_noise)

        return x.clamp(0, 1)


class RandomRingFilter:
    def __init__(self, prob: float = 0.3, kernel_size: Union[Tuple[int, int], int] = 5,
                 sigma: Union[Tuple[float, float], float] = (2.0, 5.0)) -> None:
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

        ret = ring_filter(x, kernel_size, sigma)

        return ret


class RandomBoxBlur:
    def __init__(self, prob: float = 0.3, radius: Union[Tuple[int, int], int] = 1) -> None:
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

        ret = box_blur(x, radius)

        return ret


class RandomGaussianBlur:
    def __init__(self, prob: float = 0.3, kernel_size: Union[Tuple[int, int], int] = 5,
                 sigma: Union[Tuple[float, float], float] = (0.1, 2.0)) -> None:
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

        ret = gaussian_blur(x, kernel_size, sigma)

        return ret


def test_images(dir):
    mode = torchvision.io.image.ImageReadMode.RGB
    image = torchvision.io.read_image(dir, mode)
    return image.float() / 255.


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, is_train, crop_height, crop_width, scale, dir):
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.is_train = is_train
        self.scale = scale

        imgs, names = read_images(dir, is_train)  # images list

        self.img_cnts = len(imgs)

        self.imgs = self.resize(imgs)

        self.transform_GT = torchvision.transforms.Compose(
            [
                torchvision.transforms.ColorJitter(contrast=(0, 2), brightness=(0, 2), hue=(-0.5, 0.5)),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.RandomVerticalFlip(p=0.5),
                RandomGaussianBlur(prob=0.3, kernel_size=(9, 23), sigma=(3.0, 5.0)),
                RandomGray(prob=0.3)
            ]
        )

        self.transform_GT_VALID = torchvision.transforms.Compose(
            [
                RandomGaussianBlur(prob=0.1, kernel_size=(9, 23), sigma=(3.0, 5.0))
            ]
        )

        self.transform_IR = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomOrder([
                    RandomRingFilter(prob=0.5, kernel_size=5, sigma=(2.0, 5.0)),
                    RandomUSM(prob=0.5, gray_prob=0.5),
                    RandomRescale(prob=0.5, scale=(1.0, 3.0)),
                    RandomUSM(prob=0.5, gray_prob=0.5),
                    RandomRingFilter(prob=0.5, kernel_size=5, sigma=(2.0, 5.0))
                ]),
                torchvision.transforms.RandomOrder([
                    RandomNoise(prob=1.0, gaussian_factor=25, poisson_factor=3, gray_prob=0.5, gaussian_prob=0.7),
                    RandomJPEGNoise(prob=1.0, jpeg_q=(55, 95))
                ]
                )
            ]
        )

        self.transform_LR = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomOrder([
                    RandomRingFilter(prob=0.3, kernel_size=5, sigma=(2.0, 5.0)),
                    RandomGaussianBlur(prob=0.3, kernel_size=3, sigma=(0.1, 0.5)),
                    RandomRescale(prob=0.3, scale=(1.0, 2.0)),
                    RandomUSM(prob=0.3, gray_prob=0.5),
                ]),
                RandomDownscale(scale_factor=self.scale),
                torchvision.transforms.RandomOrder([
                    RandomNoise(prob=0.5, gaussian_factor=10, poisson_factor=1, gray_prob=0.5, gaussian_prob=1.0),
                    RandomJPEGNoise(prob=0.5, jpeg_q=(75, 95))
                ])
            ]
        )

        self.transform_NR = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomOrder([
                    RandomRingFilter(prob=0.1, kernel_size=5, sigma=(2.0, 5.0)),
                    RandomGaussianBlur(prob=0.1, kernel_size=3, sigma=(0.1, 0.5)),
                    RandomUSM(prob=0.1, gray_prob=0.5),
                    RandomNoise(prob=1.0, gaussian_factor=30, poisson_factor=3, gray_prob=0.5, gaussian_prob=0.7),
                    RandomJPEGNoise(prob=1.0, jpeg_q=(35, 95))
                ])
            ]
        )

        self.transform_IR_VALID = torchvision.transforms.Compose(
            [
                RandomRescale(0.5, scale=(1.0, 2.0)),
                RandomNoise(prob=1.0, gaussian_factor=5, poisson_factor=1, gray_prob=0.5, gaussian_prob=1.0),
                RandomJPEGNoise(prob=1.0, jpeg_q=95)
            ]
        )

        self.transform_LR_VALID = torchvision.transforms.Compose(
            [
                RandomDownscale(scale_factor=self.scale),
                RandomNoise(prob=1.0, gaussian_factor=5, poisson_factor=1, gray_prob=0.5, gaussian_prob=1.0),
                RandomJPEGNoise(prob=1.0, jpeg_q=95)
            ]
        )

        self.transform_NR_VALID = torchvision.transforms.Compose(
            [
                RandomNoise(prob=1.0, gaussian_factor=5, poisson_factor=1, gray_prob=0.5, gaussian_prob=1.0),
                RandomJPEGNoise(prob=1.0, jpeg_q=75)
            ]
        )

        print('read ' + str(self.img_cnts) + ' pictures')
        print('read ' + str(len(self.imgs)) + ' examples')

    def resize(self, imgs):
        ret_list = []
        for img in imgs:
            ret_list.append(self.rand_crop(img, self.crop_height, self.crop_width))
        return ret_list

    def rand_crop(self, img, height, width):
        rect = torchvision.transforms.RandomCrop.get_params(
            img, (height, width))
        img = torchvision.transforms.functional.crop(img, *rect)
        return img

    def __getitem__(self, idx):
        if self.is_train:
            x = self.imgs[idx]
            x = x.float() / 255.
            x = self.transform_GT(x)
            lr, hr = x, x

            if self.scale <= 1:
                # 图像恢复
                lr = self.transform_IR(lr)
            else:
                # 图像超分
                lr = self.transform_LR(lr)

        else:
            x = self.imgs[idx]
            x = x.float() / 255.
            x = self.transform_GT_VALID(x)

            lr, hr = x, x

            if self.scale <= 1:
                lr = self.transform_IR_VALID(lr)
            else:
                lr = self.transform_LR_VALID(lr)

        return lr, hr

    def __len__(self):
        return len(self.imgs)


def load_data(batch_size, crop_height, crop_width, scale):
    dir = 'E:/work/AIR'
    num_workers = 4
    train_iter = torch.utils.data.DataLoader(
        TrainDataset(True, crop_height, crop_width, scale, dir), batch_size,
        shuffle=True, drop_last=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(
        TrainDataset(False, crop_height, crop_width, scale, dir), batch_size,
        drop_last=True, num_workers=num_workers)
    return train_iter, test_iter
