import random
from utils.logger import logger
import numpy as np
import torch
import torchvision
import torchvision.transforms as TT
from torchvision.transforms import functional as TTF
from torch.nn import functional as TF
from torch.utils import data
import os
import math
import cv2
from typing import Any, Mapping, Optional, Sequence, Union, Tuple
from wand.image import Image, IMAGE_TYPES
import wand
from scipy import special
from PIL import Image as pim

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
np.seterr(invalid='ignore')


def read_images(is_train: bool = True) -> Tuple[Sequence[torch.Tensor], Sequence[str]]:
    mode = torchvision.io.image.ImageReadMode.RGB
    labels = []
    filenames = []
    #dataset =  r"E:\Encode\Dataset\EAIRDMS_PLUS"  # "final_data"#
    dataset = r"E:\Encode\Dataset\AA"
    if is_train:
        train_list = os.listdir(os.path.join(dataset, "train"))
        for train_img in train_list:
            if train_img.split(".").pop() in ["png"]:
                img = torchvision.io.read_image(os.path.join(dataset, "train", train_img), mode)
                labels.append(img)
                filenames.append(train_img)

    else:
        valid_list = os.listdir(os.path.join(dataset, "valid"))
        for valid_img in valid_list:
            if valid_img.split(".").pop() in ["png"]:
                img = torchvision.io.read_image(os.path.join(dataset, "valid", valid_img), mode)
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
    ret = Image.from_array(arr, channel_map=channel_map, storage=storage)
    if channel_map == "I":
        ret.type = "grayscale"
    return ret


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


'''
def repair(flt, src):
    if len(flt.shape) == 2:
        flt = flt.unsqueeze(0)

    if len(src.shape) == 2:
        src = src.unsqueeze(0)

    channel, height, width = src.shape

    src_pad = F.pad(src, (1, 1, 1, 1), mode="constant")

    for i, c in enumerate(src_pad):
        m1 = c[0:height, 0:width]
        m2 = c[0:height, 1:width + 1]
        m3 = c[0:height, 2:width + 2]
        m4 = c[1:height + 1, 0:width]
        m5 = c[1:height + 1, 1:width + 1]
        m6 = c[1:height + 1, 2:width + 2]
        m7 = c[2:height + 2, 0:width]
        m8 = c[2:height + 2, 1:width + 1]
        m9 = c[2:height + 2, 2:width + 2]
        mstack = torch.stack([m1, m2, m3, m4, m5, m6, m7, m8, m9], dim=0)
        stack_max, _ = torch.max(mstack, dim=0)
        stack_min, _ = torch.min(mstack, dim=0)
        cp_max, _ = torch.max(torch.stack([flt[i], stack_min], dim=0), dim=0)
        cp_res, _ = torch.min(torch.stack([cp_max, stack_max], dim=0), dim=0)
        flt[i] = cp_res

    return flt
'''


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

    y: torch.Tensor = (0.299 * r + 0.587 * g + 0.114 * b).clamp(0, 1)
    u: torch.Tensor = (-0.147 * r - 0.289 * g + 0.436 * b).clamp(-0.5, 0.5)
    v: torch.Tensor = (0.615 * r - 0.515 * g - 0.100 * b).clamp(-0.5, 0.5)

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

    return out.clamp(0, 1)


def filter2D(img: torch.Tensor, kernel_size: int, kernel: torch.Tensor) -> torch.Tensor:
    dim = len(img.shape)

    if dim == 2:
        img = img.unsqueeze(0).unsqueeze(0)

    elif dim == 3:
        img = img.unsqueeze(0)

    b, c, h, w = img.shape

    img_pad = TF.pad(img, (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2), mode="reflect")

    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(c, 1, 1, 1)

    out = TF.conv2d(img_pad, weight=kernel, bias=None, stride=1, groups=c)

    if dim == 2:
        out = out.squeeze(0).squeeze(0)
    elif dim == 3:
        out = out.squeeze(0)

    return out


def rescale(x: torch.Tensor, scale: float = 1.0, anisotropic_p: float = 0.3,
            anisotrpoic_scale: float = 0.25) -> torch.Tensor:
    if len(x.shape) <= 2:
        x = x.unsqueeze(0)
        g = True
    else:
        g = False

    sw = x.shape[2]
    sh = x.shape[1]

    kernel_box = ["box"]
    kernel_bilinear = ["triangle"]
    kernel_bicubic = ["hermite", "catrom", "mitchell", "robidoux", "spline"]
    kernel_window = ["lanczos", "jinc", "sinc"]

    kernel_type = [kernel_box, kernel_bilinear, kernel_bicubic, kernel_window]
    kernel_type_wo_box = [kernel_bilinear, kernel_bicubic, kernel_window]

    kernel_type_weight = [1 / 2, 1 / 2, 1, 1]
    kernel_type_weight_wo_box = [1 / 2, 1, 1]

    kernel_type_choice = random.choices(kernel_type, weights=kernel_type_weight, k=2)
    kernel_type_choice_wo_box = random.choices(kernel_type_wo_box, weights=kernel_type_weight_wo_box, k=2)

    interpolation_kernels_up = random.choice(kernel_type_choice[0])
    interpolation_kernels_down = random.choice(kernel_type_choice[1])
    interpolation_kernels_pre_box = random.choice(kernel_type_choice_wo_box[0])
    interpolation_kernels_back = random.choice(kernel_type_choice_wo_box[1])

    min_l = math.floor(sw * 0.375)
    max_l = math.floor(sw * 0.5)

    h_l = sw * 0.5

    dw = random.randint(min_l, max_l)
    dh = dw

    if interpolation_kernels_down == "box":
        if dh == h_l and dw == h_l:
            x = resize(x, dw, dh, "box")
        elif min(dw, dh) > h_l:
            x = resize(x, dw * 2, dh * 2, interpolation_kernels_pre_box)
            x = resize(x, dw, dh, "box")
        else:
            x = resize(x, dw * 3, dh * 3, interpolation_kernels_pre_box, )
            x = resize(x, dw, dh, "box")
    else:
        x = resize(x, dw, dh, interpolation_kernels_down)

    if interpolation_kernels_up == "box":
        if dh == h_l and dw == h_l:
            x = resize(x, sw, sh, "box")
        elif min(dw, dh) > h_l:
            x = resize(x, dw * 2, dh * 2, "box")
            x = resize(x, sw, sh, interpolation_kernels_back)
        else:
            x = resize(x, dw * 3, dh * 3, "box")
            x = resize(x, sw, sh, interpolation_kernels_back)
    else:
        x = resize(x, sw, sh, interpolation_kernels_up)

    # print(x.shape)

    if g:
        x = x.squeeze(0)

    # print(x.shape)
    return x


def gaussian_noise(img: torch.Tensor, sigma: float, amount: float = 1.0, gray_noise: bool = False,
                   blur_noise: bool = False) -> torch.Tensor:
    img_yuv = rgb_to_yuv(img)
    y = img_yuv[..., 0, :, :]
    u = img_yuv[..., 1, :, :]
    v = img_yuv[..., 2, :, :]
    if gray_noise:
        imshape = y.shape
    else:
        imshape = img.shape

    rg = torch.normal(mean=0, std=sigma, size=imshape)

    # if random.uniform(0, 1) < 0.5:
    #    rg = rg * amount

    if blur_noise:
        #rg = rg + 0.5
        rg = rescale(rg)
        #rg = rg - 0.5
    else:
        pass

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


def crop(img: torch.Tensor, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0) -> torch.Tensor:
    c, h, w = img.shape
    ret = TTF.crop(img, left=left, top=top, height=h - top - bottom,
                                                 width=w - left - right)
    return ret


def add_border(img: torch.Tensor, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0) -> torch.Tensor:
    ret = TTF.pad(img, padding=[left, top, right, bottom], padding_mode="constant")
    return ret


def box_blur(img: torch.Tensor, radius: int = 1) -> torch.Tensor:
    ksize = 2 * radius + 1

    k = torch.ones((ksize, ksize), dtype=torch.float32)
    k = k / k.sum()

    out = filter2D(img, ksize, k)
    return out.clamp(0, 1)


def mod_blur(img: torch.Tensor) -> torch.Tensor:
    ksize = 3

    parms = random.sample(range(1, 10), k=3)
    parms.sort()
    parm1 = parms[0]
    parm2 = parms[1]
    parm3 = parms[2]

    k = torch.tensor([[parm1, parm2, parm1], [parm2, parm3, parm2], [parm1, parm2, parm1]], dtype=torch.float32)
    k = k / k.sum()

    out = filter2D(img, ksize, k)
    return out.clamp(0, 1)


def unsharp_mask(img: torch.Tensor, blur: str = "gaussian", kernel_size: int = 3, sigma: float = 2.0, gray: bool = True,
                 radius: int = 1, amount: float = 1.0) -> torch.Tensor:
    if blur == "gaussian":
        nr = gaussian_blur(img, kernel_size, sigma).clamp(0, 1)
    elif blur == "box":
        nr = box_blur(img, radius)
    else:
        raise ValueError("未知blur")

    makediff = (img - nr) * amount
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

    out = filter2D(img, ksize, h)

    return out.clamp(0, 1)


def lanczos_filter(img: torch.Tensor, ksize: int, sigma: float) -> torch.Tensor:
    n = (ksize - 1.0) / 2.0
    x = torch.arange(-n, n + 1, dtype=torch.float32).reshape(ksize, 1)
    y = torch.arange(-n, n + 1, dtype=torch.float32).reshape(1, ksize)

    d = torch.sqrt((x * x) + (y * y))
    h = torch.sinc(d) * torch.sinc(d / sigma)
    h = torch.div(h, h.sum())

    out = filter2D(img, ksize, h)
    return out.clamp(0, 1)


def sinc_filter(img: torch.Tensor, ksize: int, cutoff: float, pad_to: int = 0) -> torch.Tensor:
    kernel = np.fromfunction(
        lambda x, y: cutoff * special.j1(cutoff * np.sqrt(
            (x - (ksize - 1) / 2) ** 2 + (y - (ksize - 1) / 2) ** 2)) / (2 * np.pi * np.sqrt(
            (x - (ksize - 1) / 2) ** 2 + (y - (ksize - 1) / 2) ** 2)), [ksize, ksize])
    kernel[(ksize - 1) // 2, (ksize - 1) // 2] = cutoff ** 2 / (4 * np.pi)
    kernel = kernel / np.sum(kernel)
    if pad_to > ksize:
        pad_size = (pad_to - ksize) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

    k = torch.from_numpy(kernel).float()
    out = filter2D(img, ksize, k)
    return out.clamp(0, 1)


def laplacian_sharpen(img: torch.Tensor, gray: bool = True) -> torch.Tensor:
    if random.uniform(0, 1) < 0.5:
        k = torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=torch.float32)
    else:
        k = torch.tensor([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=torch.float32)

    if gray:
        img_yuv = rgb_to_yuv(img)
        y = img_yuv[..., 0, :, :]
        u = img_yuv[..., 1, :, :]
        v = img_yuv[..., 2, :, :]
        y = filter2D(y, 3, k)
        out = yuv_to_rgb(torch.stack([y, u, v], -3))
    else:
        out = filter2D(img, 3, k)

    return out.clamp(0, 1)


def sharpen(img: torch.Tensor, amount: float = 1.0, gray: bool = True) -> torch.Tensor:
    kx = torch.tensor([(1 - 2 ** amount) / 2, 2 ** amount, (1 - 2 ** amount) / 2], dtype=torch.float32).reshape(3, 1)
    ky = torch.tensor([(1 - 2 ** amount) / 2, 2 ** amount, (1 - 2 ** amount) / 2], dtype=torch.float32).reshape(1, 3)
    k = kx * ky

    if gray:
        img_yuv = rgb_to_yuv(img)
        y = img_yuv[..., 0, :, :]
        u = img_yuv[..., 1, :, :]
        v = img_yuv[..., 2, :, :]
        y = filter2D(y, 3, k)
        out = yuv_to_rgb(torch.stack([y, u, v], -3))
    else:
        out = filter2D(img, 3, k)

    return out.clamp(0, 1)


def choose_jpeg_quality(noise_level):
    qualities = []
    if noise_level == 0:
        qualities.append(random.randint(85, 95))
    elif noise_level == 1:
        qualities.append(random.randint(65, 85))
    elif noise_level in {2, 3}:
        # 2 and 3 are the same, NR_RATE is different
        r = random.uniform(0, 1)
        if r > 0.4:
            qualities.append(random.randint(25, 70))
        elif r > 0.1:
            # nunif: Add high quality patterns
            if random.uniform(0, 1) < 0.05:
                quality1 = random.randint(35, 95)
            else:
                quality1 = random.randint(35, 70)
            quality2 = quality1 - random.randint(5, 10)
            qualities.append(quality1)
            qualities.append(quality2)
        else:
            # nunif: Add high quality patterns
            if random.uniform(0, 1) < 0.05:
                quality1 = random.randint(50, 95)
            else:
                quality1 = random.randint(50, 70)
            quality2 = quality1 - random.randint(5, 15)
            quality3 = quality1 - random.randint(15, 25)
            qualities.append(quality1)
            qualities.append(quality2)
            qualities.append(quality3)
    return qualities


def limit_number(num, min, max):
    if num < min:
        num = min
    if num > max:
        num = max
    return num


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
            chroma_subsampling = True
        else:
            chroma_subsampling = False

        x = jpeg_compression(x, jpeg_qualities, chroma_subsampling)

        return x


class RandomBlock:
    def __init__(self, prob: float = 1.0, noise_level: int = 3,
                 css_prob: float = 0.5) -> None:
        self.prob = prob
        self.css_prob = css_prob
        self.noise_level = noise_level

    def __call__(self, x) -> torch.Tensor:
        if random.uniform(0, 1) > self.prob:
            return x

        if (self.noise_level == 3 and random.uniform(0, 1) < 0.95) or random.uniform(0, 1) < 0.75:
            # use noise_level noise
            qualities = choose_jpeg_quality(self.noise_level)
        else:
            noise_level = random.randint(0, self.noise_level)
            qualities = choose_jpeg_quality(noise_level)

        if random.uniform(0, 1) < self.css_prob:
            chroma_subsampling = True
        else:
            chroma_subsampling = False

        for jpeg_quality in qualities:
            x = jpeg_compression(x, jpeg_quality, chroma_subsampling)
        return x


class RandomDownscale:
    def __init__(self, scale_factor: int = 2) -> None:
        self.scale_factor = scale_factor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        sw = x.shape[2]
        sh = x.shape[1]

        kernel_box = ["box"]
        kernel_bilinear = ["triangle"]
        kernel_bicubic = ["hermite", "catrom", "mitchell", "robidoux", "spline"]
        kernel_window = ["lanczos", "jinc", "sinc"]

        kernel_type = [kernel_box, kernel_bilinear, kernel_bicubic, kernel_window]

        kernel_type_weight = [1 / 2, 1 / 2, 1, 1]
        kernel_type_choice = random.choices(kernel_type, weights=kernel_type_weight, k=1)

        kernel = random.choice(kernel_type_choice[0])
        x = resize(x, sw // self.scale_factor, sh // self.scale_factor, kernel)

        return x


class RandomRescale:
    def __init__(self, prob: float = 0.3, scale: Tuple[float, float] = (0.3, 1.0), anisotropic_p: float = 0.3,
                 anisotrpoic_scale: float = 0.25) -> None:
        self.prob = prob
        self.scale_range = scale
        # self.weight = weight
        self.anisotropic_p = anisotropic_p
        self.anisotropic_scale = anisotrpoic_scale

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.uniform(0, 1) > self.prob:
            return x
        sw = x.shape[2]
        sh = x.shape[1]

        # print(sw,sh)

        kernel_box = ["box"]
        kernel_bilinear = ["triangle"]
        kernel_bicubic = ["hermite", "catrom", "mitchell", "robidoux", "robidouxsharp", "lagrange", "spline"]
        kernel_window = ["lanczos", "lanczossharp", "lanczos2", "lanczos2sharp", "jinc", "sinc", "hanning", "hamming",
                         "blackman", "kaiser", "welsh", "parzen", "bohman", "bartlett", "cosine", "sentinel"]

        kernel_type = [kernel_box, kernel_bilinear, kernel_bicubic, kernel_window]
        kernel_type_wo_box = [kernel_bilinear, kernel_bicubic, kernel_window]

        kernel_type_weight = [1, 1 / 2, 1, 1]
        kernel_type_weight_wo_box = [1 / 2, 1, 1]

        kernel_type_choice = random.choices(kernel_type, weights=kernel_type_weight, k=2)
        kernel_type_choice_wo_box = random.choices(kernel_type_wo_box, weights=kernel_type_weight_wo_box, k=2)

        interpolation_kernels_up = random.choice(kernel_type_choice[0])
        interpolation_kernels_down = random.choice(kernel_type_choice[1])
        interpolation_kernels_pre_box = random.choice(kernel_type_choice_wo_box[0])
        interpolation_kernels_back = random.choice(kernel_type_choice_wo_box[1])

        if sw == 128:
            min_l = 48
            max_l = 128
            anisotropic_l = 32
            h_l = 64
            t_l = 42
        elif sw == 144:
            min_l = 54
            max_l = 144
            anisotropic_l = 36
            h_l = 72
            t_l = 48

        elif sw == 64:
            min_l = 24
            max_l = 64
            anisotropic_l = 16
            h_l = 32
            t_l = 20

        elif sw == 80:
            min_l = 30
            max_l = 80
            anisotropic_l = 20
            h_l = 40
            t_l = 24

        else:
            raise ValueError("sw error")

        if random.uniform(0, 1) < self.anisotropic_p:
            scale_shift = random.randint(0, anisotropic_l)
            dw = random.randint(min_l, max_l)
            dh = dw
            if random.uniform(0, 1) < 0.5:
                dh += scale_shift
            else:
                dh -= scale_shift

            dh = limit_number(dh, min_l, max_l)
        else:
            dw = random.randint(min_l, max_l)
            dh = dw

        if random.uniform(0, 1) < 0.5:
            dw, dh = dh, dw

        if interpolation_kernels_down == "box":
            if (dh == h_l and dw == h_l) or (dh == t_l and dw == t_l):
                x = resize(x, dw, dh, "box")
            elif min(dw, dh) > h_l:
                x = resize(x, dw * 2, dh * 2, interpolation_kernels_pre_box)
                x = resize(x, dw, dh, "box")
            else:
                x = resize(x, dw * 3, dh * 3, interpolation_kernels_pre_box)
                x = resize(x, dw, dh, "box")
        else:
            x = resize(x, dw, dh, interpolation_kernels_down)

        if interpolation_kernels_up == "box":
            if (dh == h_l and dw == h_l) or (dh == t_l and dw == t_l):
                x = resize(x, sw, sh, "box")
            elif min(dw, dh) > h_l:
                x = resize(x, dw * 2, dh * 2, "box")
                x = resize(x, sw, sh, interpolation_kernels_back)
            else:
                x = resize(x, dw * 3, dh * 3, "box")
                x = resize(x, sw, sh, interpolation_kernels_back)
        else:
            x = resize(x, sw, sh, interpolation_kernels_up)

        return x.clamp(0, 1)


class RandomRescale2:
    def __init__(self, prob: float = 0.3, scale: Tuple[float, float] = (0.3, 1.0), anisotropic_p: float = 0.3,
                 anisotrpoic_scale: float = 0.25) -> None:
        self.prob = prob
        self.scale_range = scale
        self.anisotropic_p = anisotropic_p
        self.anisotropic_scale = anisotrpoic_scale

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.uniform(0, 1) > self.prob:
            return x
        sw = x.shape[2]
        sh = x.shape[1]

        kernel_box = ["box"]
        kernel_bilinear = ["triangle"]
        kernel_bicubic = ["hermite", "catrom", "mitchell", "robidoux", "robidouxsharp", "lagrange"]
        kernel_window = ["lanczos", "lanczossharp", "lanczos2", "lanczos2sharp", "jinc", "sinc", "hanning", "hamming",
                         "blackman", "kaiser", "welsh", "parzen", "bohman", "bartlett", "cosine", "sentinel"]

        kernel_type = [kernel_box, kernel_bilinear, kernel_bicubic, kernel_window]
        kernel_type_wo_box = [kernel_bilinear, kernel_bicubic, kernel_window]

        kernel_type_weight = [1, 1 / 2, 1, 1]
        kernel_type_weight_wo_box = [1 / 2, 1, 1]

        kernel_type_choice = random.choices(kernel_type, weights=kernel_type_weight, k=2)
        kernel_type_choice_wo_box = random.choices(kernel_type_wo_box, weights=kernel_type_weight_wo_box, k=2)

        interpolation_kernels_up = random.choice(kernel_type_choice[0])
        interpolation_kernels_down = random.choice(kernel_type_choice[1])
        interpolation_kernels_pre_box = random.choice(kernel_type_choice_wo_box[0])
        interpolation_kernels_back = random.choice(kernel_type_choice_wo_box[1])

        if sw == 128:
            min_l = 80
            max_l = 128
            anisotropic_l = 32
            h_l = 64
            t_l = 42
        elif sw == 144:
            min_l = 90
            max_l = 144
            anisotropic_l = 36
            h_l = 72
            t_l = 48

        elif sw == 64:
            min_l = 40
            max_l = 64
            anisotropic_l = 16
            h_l = 32
            t_l = 20

        elif sw == 80:
            min_l = 50
            max_l = 80
            anisotropic_l = 20
            h_l = 40
            t_l = 24

        else:
            raise ValueError("sw error")

        if random.uniform(0, 1) < self.anisotropic_p:
            scale_shift = random.randint(0, anisotropic_l)
            dw = random.randint(min_l, max_l)
            dh = dw
            if random.uniform(0, 1) < 0.5:
                dh += scale_shift
            else:
                dh -= scale_shift

            dh = limit_number(dh, min_l, max_l)
        else:
            dw = random.randint(min_l, max_l)
            dh = dw

        if random.uniform(0, 1) < 0.5:
            dw, dh = dh, dw

        if interpolation_kernels_down == "box":
            if (dh == h_l and dw == h_l) or (dh == t_l and dw == t_l):
                x = resize(x, dw, dh, "box")
            elif min(dw, dh) > h_l:
                x = resize(x, dw * 2, dh * 2, interpolation_kernels_pre_box)
                x = resize(x, dw, dh, "box")
            else:
                x = resize(x, dw * 3, dh * 3, interpolation_kernels_pre_box)
                x = resize(x, dw, dh, "box")
        else:
            x = resize(x, dw, dh, interpolation_kernels_down)

        if interpolation_kernels_up == "box":
            if (dh == h_l and dw == h_l) or (dh == t_l and dw == t_l):
                x = resize(x, sw, sh, "box")
            elif min(dw, dh) > h_l:
                x = resize(x, dw * 2, dh * 2, "box")
                x = resize(x, sw, sh, interpolation_kernels_back)
            else:
                x = resize(x, dw * 3, dh * 3, "box")
                x = resize(x, sw, sh, interpolation_kernels_back)
        else:
            x = resize(x, sw, sh, interpolation_kernels_up)

        return x.clamp(0, 1)


class RandomAA:
    def __init__(self, prob: float = 0.3, anisotropic_p: float = 0.3) -> None:
        self.prob = prob
        self.anisotropic_p = anisotropic_p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.uniform(0, 1) > self.prob:
            return x
        sw = x.shape[2]
        sh = x.shape[1]

        kernel_box = ["box"]
        kernel_bicubic = ["catrom"]
        kernel_window = ["lanczos", "lanczossharp", "lanczos2", "lanczos2sharp", "jinc", "sinc", "hanning", "hamming",
                         "blackman", "kaiser", "welsh", "parzen", "bohman", "bartlett", "cosine", "sentinel"]

        kernel_type_weight = [1, 1, 1]
        kernel_type_weight_wo_box = [1, 1]

        kernel_type = [kernel_box, kernel_bicubic, kernel_window]
        kernel_type_wo_box = [kernel_bicubic, kernel_window]

        kernel_type_choice = random.choices(kernel_type, weights=kernel_type_weight, k=2)
        kernel_type_choice_wo_box = random.choices(kernel_type_wo_box, weights=kernel_type_weight_wo_box, k=2)

        if sw == 128:
            min_l = 128
            max_l = 320
            anisotropic_l = 128
            d_l = 256
        elif sw == 144:
            min_l = 144
            max_l = 360
            anisotropic_l = 144
            d_l = 288
        elif sw == 64:
            min_l = 64
            max_l = 160
            anisotropic_l = 64
            d_l = 128
        elif sw == 80:
            min_l = 80
            max_l = 200
            anisotropic_l = 80
            d_l = 160

        else:
            raise ValueError("sw error")

        if random.uniform(0, 1) < self.anisotropic_p:
            scale_shift = random.randint(0, anisotropic_l)
            dw = random.randint(min_l, max_l)
            dh = dw
            if random.uniform(0, 1) < 0.5:
                dh += scale_shift
            else:
                dh -= scale_shift

            dh = limit_number(dh, min_l, max_l)
        else:
            dw = random.randint(min_l, max_l)
            dh = dw

        interpolation_kernels_up = random.choice(kernel_type_choice[0])
        interpolation_kernels_down = random.choice(kernel_type_choice[1])
        interpolation_kernels_post_box = random.choice(kernel_type_choice_wo_box[0])
        interpolation_kernels_back = random.choice(kernel_type_choice_wo_box[1])

        if interpolation_kernels_up == "box":
            if dh == d_l and dw == d_l:
                x = resize(x, dw, dh, "box")
            elif max(dh, dw) < d_l:
                x = resize(x, sw * 2, sh * 2, "box")
                x = resize(x, dw, dh, interpolation_kernels_post_box)
            else:
                x = resize(x, sw * 3, sh * 3, "box")
                x = resize(x, dw, dh, interpolation_kernels_post_box)
        else:
            x = resize(x, dw, dh, interpolation_kernels_up)

        if interpolation_kernels_down == "box":
            if dh == d_l and dw == d_l:
                x = resize(x, sw, sw, "box")
            elif max(dh, dw) < d_l:
                x = resize(x, sw * 2, sh * 2, interpolation_kernels_back)
                x = resize(x, sw, sh, "box")
            else:
                x = resize(x, sw * 3, sh * 3, interpolation_kernels_back)
                x = resize(x, sw, sh, "box")
        else:
            x = resize(x, sw, sh, interpolation_kernels_down)

        # print("dw: ",dw," sw: ",sw)

        return x.clamp(0, 1)


class RandomGray:
    def __init__(self, prob: float = 0.3) -> None:
        self.prob = prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.uniform(0, 1) > self.prob:
            return x
        return torchvision.transforms.functional.rgb_to_grayscale(x, 3)


class RandomSharpen:
    def __init__(self, prob: float = 0.3, sigma: Union[Tuple[float, float], float] = (2.0, 5.0),
                 radius: Union[Tuple[int, int], int] = (1, 2), gray_prob: float = 0.5) -> None:
        self.prob = prob
        self.gray_prob = gray_prob
        self.sigma = sigma
        self.radius = radius

        self.kernel_sizes = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
        self.kernel_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        self.blur_type = ["gaussian", "box"]

        self.sharpen_type = ["usm", "laplacian", "avs"]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.uniform(0, 1) > self.prob:
            return x

        if random.uniform(0, 1) > self.gray_prob:
            gray = False
        else:
            gray = True

        sp_type = random.choice(self.sharpen_type)

        if sp_type == "usm":
            sigma = random.uniform(self.sigma[0], self.sigma[1])

            radius = random.randint(self.radius[0], self.radius[1])

            ksize = random.choices(self.kernel_sizes, weights=self.kernel_weights, k=1)[0]

            amount = 1
            if random.uniform(0, 1) < 0.3:
                amount = random.uniform(0.5, 2.0)

            blur_type = random.choice(self.blur_type)

            sharp = unsharp_mask(x, blur_type, ksize, sigma, gray, radius, amount)
        elif sp_type == "laplacian":
            sharp = laplacian_sharpen(x, gray)
        else:
            amount = random.uniform(0, 1)
            sharp = sharpen(x, amount, gray)

        return sharp


class RandomNoise:
    def __init__(self, prob: float = 0.65, gaussian_factor: int = 25, gray_prob: float = 0.5,
                 blur_prob: float = 0.3) -> None:
        self.prob = prob
        self.gaussian_factor = gaussian_factor
        self.gray_prob = gray_prob
        self.blur_prob = blur_prob

    def __call__(self, x) -> torch.Tensor:
        if random.uniform(0, 1) > self.prob:
            return x

        if random.uniform(0, 1) < self.gray_prob:
            gray_noise = True
        else:
            gray_noise = False

        if random.uniform(0, 1) < self.blur_prob:
            blur_noise = True
        else:
            blur_noise = False

        sigma = random.randint(0, self.gaussian_factor)

        amount = 1

        x = gaussian_noise(x, sigma / 255., amount, gray_noise, blur_noise)

        return x.clamp(0, 1)


class RandomLanczosFilter:
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

        ret = lanczos_filter(x, kernel_size, sigma)
        return ret


class RandomSincFilter:
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

        ret = sinc_filter(x, kernel_size, sigma, 0)

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

        if random.uniform(0, 1) < 0.3:
            ret = box_blur(x, radius)
        else:
            ret = mod_blur(x)

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

        ret = gaussian_blur(x, kernel_size, sigma).clamp(0, 1)

        return ret


class RandomBlur:
    def __init__(self, prob: float = 0.3, kernel_size: Union[Tuple[int, int], int] = 5,
                 sigma: Union[Tuple[float, float], float] = (0.1, 1.0),
                 radius: Union[Tuple[int, int], int] = 1) -> None:
        self.prob = prob
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.radius = radius

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
        if random.uniform(0, 1) < 0.5:
            if random.uniform(0, 1) < 0.5:
                ret = box_blur(x, radius)
            else:
                ret = mod_blur(x)
        else:
            ret = gaussian_blur(x, kernel_size, sigma).clamp(0, 1)

        return ret

class RandomBlend:
    def __init__(self, prob: float = 0.3, color_prob:float=0.3) -> None:
        self.prob = prob
        self.color_prob = color_prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.uniform(0, 1) > self.prob:
            return x
        c, h, w = x.shape
        x = TTF.to_pil_image(x)
        alpha = random.uniform(0,1)
        if random.uniform(0,1)<self.color_prob:
            gray = random.randint(0,255)
            bg_color = (gray,gray,gray)
        else:
            r = random.randint(0,255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            bg_color = (r,g,b)

        bg = pim.new("RGB", (h, w), bg_color)

        ret = pim.blend(x,bg,alpha)
        ret = TTF.to_tensor(ret)
        return ret

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
        rot = TTF.rotate(pad, angle=ang,
                                                       interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
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
        rot = TTF.rotate(x, angle=ang,
                                                       interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
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


def test_images(dir):
    mode = torchvision.io.image.ImageReadMode.RGB
    image = torchvision.io.read_image(dir, mode)
    return image.float() / 255.


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, is_train, crop_height, crop_width, scale):
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.is_train = is_train
        self.scale = scale

        self.images, self.names,= read_images(is_train)  # images list

        self.images_counts = len(self.images)


        self.transform_GT = TT.Compose(
            [
                TT.RandomInvert(0.5),
                TT.RandomApply([TT.ColorJitter(brightness=(0, 2), contrast=(0, 2), saturation=(0, 2), hue=(-0.5, 0.5))],
                               p=0.5),
                TT.RandomHorizontalFlip(p=0.5),
                TT.RandomVerticalFlip(p=0.5),
                #RandomRotate(prob=0.5),
                RandomBlur(prob=0.3, kernel_size=(9, 23), sigma=(2.5, 5.0), radius=(2, 5)),
                RandomGray(prob=0.3)
            ]
        )

        self.transform_GT_VALID = TT.Compose(
            [
                RandomGaussianBlur(prob=0.1, kernel_size=(9, 23), sigma=(3.0, 5.0))
            ]
        )



        self.transform_IR = TT.Compose(
            [

                TT.RandomOrder([
                    TT.RandomChoice([
                        RandomLanczosFilter(prob=0.4, kernel_size=(7, 21), sigma=(2.0, 5.0)),
                        RandomSincFilter(prob=0.4, kernel_size=(3, 21), sigma=(2.0, np.pi)),
                    ]),
                    RandomSharpen(prob=0.4, gray_prob=0.5, sigma=(0.2, 2.0)),
                    RandomRescale(prob=0.8, scale=(0.375, 1.0), anisotropic_p=0.3, anisotrpoic_scale=0.25),
                    RandomGaussianBlur(prob=0.4,kernel_size=(3,21),sigma=(0.2,0.5)),
                    TT.RandomChoice([
                        RandomRescale2(prob=0.4, scale=(0.375, 1.0), anisotropic_p=0.3, anisotrpoic_scale=0.25),
                        RandomBlur(prob=0.4, kernel_size=(3, 21), sigma=(0.1, 1.0), radius=1)
                    ]),
                    RandomJPEGNoise(prob=0.4, jpeg_q=(95, 100), css_prob=1.0)
                ]),
                TT.RandomOrder([
                    RandomNoise(prob=0.4, gaussian_factor=25, gray_prob=0.5, blur_prob=0.1),
                    RandomJPEGNoise(prob=0.4, jpeg_q=(25, 95), css_prob=0.0)
                ])
            ]
        )

        self.transform_IR_WOSCREENTONE = TT.Compose(
            [

                TT.RandomOrder([
                    TT.RandomChoice([
                        RandomLanczosFilter(prob=0.4, kernel_size=(7, 21), sigma=(2.0, 5.0)),
                        RandomSincFilter(prob=0.4, kernel_size=(3, 21), sigma=(2.0, np.pi)),
                    ]),
                    RandomSharpen(prob=0.4, gray_prob=0.5, sigma=(0.2, 2.0)),
                    RandomRescale2(prob=0.8, scale=(0.375, 1.0), anisotropic_p=0.3, anisotrpoic_scale=0.25),
                    RandomGaussianBlur(prob=0.4, kernel_size=(3, 21), sigma=(0.2, 0.5)),
                    TT.RandomChoice([
                        RandomRescale2(prob=0.4, scale=(0.375, 1.0), anisotropic_p=0.3, anisotrpoic_scale=0.25),
                        RandomBlur(prob=0.4, kernel_size=(3, 21), sigma=(0.1, 1.0), radius=1)
                    ]),
                    RandomJPEGNoise(prob=0.4, jpeg_q=(95, 100), css_prob=1.0)
                ]),
                TT.RandomOrder([
                    RandomNoise(prob=0.4, gaussian_factor=25, gray_prob=0.5, blur_prob=0.1),
                    RandomJPEGNoise(prob=0.4, jpeg_q=(25, 95), css_prob=0.0)
                ])
            ]
        )

        self.transform_EDGE = TT.Compose(
            [
                TT.RandomOrder([
                    RamdomAugBorder(prob=0.8),
                    RandomNoise(prob=0.4, gaussian_factor=25, gray_prob=0.5, blur_prob=0.3),
                    RandomJPEGNoise(prob=0.6, jpeg_q=(25, 95), css_prob=0.5)
                ])

            ]
        )

        self.transform_LR = TT.Compose(
            [
                TT.RandomOrder([
                    TT.RandomChoice([
                        RandomLanczosFilter(prob=0.4, kernel_size=(7, 21), sigma=(2.0, 5.0)),
                        RandomSincFilter(prob=0.4, kernel_size=(3, 21), sigma=(2.0, np.pi)),
                    ]),
                    RandomSharpen(prob=0.4, gray_prob=0.5, sigma=(0.2, 2.0)),
                    RandomRescale(prob=0.8, scale=(0.375, 1.0), anisotropic_p=0.3, anisotrpoic_scale=0.25),
                    RandomAA(prob=0.6),
                    TT.RandomChoice([
                        RandomRescale2(prob=0.5, scale=(0.375, 1.0), anisotropic_p=0.3, anisotrpoic_scale=0.25),
                        RandomBlur(prob=0.5, kernel_size=(3, 21), sigma=(0.1, 1.0), radius=1)
                    ]),
                    RandomJPEGNoise(prob=0.5, jpeg_q=(95, 100), css_prob=1.0)
                ]),
            ]
        )

        self.transform_pad = TT.Pad(padding=8)

        self.transform_LR_TEST_V = TT.Compose(
            [
                RandomDownscale(scale_factor=self.scale),
                RandomJPEGNoise(prob=0.7, jpeg_q=95, css_prob=0.5)
            ]
        )

        self.transform_IR_VALID = TT.Compose(
            [
                RandomRescale(0.5, scale=(1.0, 2.0)),
                RandomJPEGNoise(prob=1.0, jpeg_q=95),
                RandomNoise(prob=1.0, gaussian_factor=5, gray_prob=0.5)
            ]
        )

        self.transform_LR_VALID = TT.Compose(
            [
                RandomDownscale(scale_factor=self.scale),
                RandomNoise(prob=1.0, gaussian_factor=5, gray_prob=0.5),
                RandomJPEGNoise(prob=1.0, jpeg_q=95)
            ]
        )

        logger.info(f"read {self.images_counts} pictures")
        logger.info(f"read {len(self.images)} examples")

    def rand_crop(self, img, height, width):
        rect = TT.RandomCrop.get_params(
            img, (height, width))
        img = TTF.crop(img, *rect)
        return img

    def center_crop(self, img, height, width):
        img = TTF.center_crop(img, (height, width))
        return img

    def __getitem__(self, idx):
        if self.is_train:
            f = 0
            x = self.images[idx]
            x = self.rand_crop(x, self.crop_height, self.crop_width)
            #x = self.center_crop(x, self.crop_height, self.crop_width)
            x = x.float() / 255.
            x = self.transform_GT(x)

            lr = x.clone()
            hr = x.clone()
            if self.scale <= 1:
                # 图像恢复
                if random.uniform(0, 1) < 0.4:
                    f = 1
                    lr = self.transform_pad(lr)
                if "SCREENTONE" in self.names:
                    lr = self.transform_IR(lr)
                else:
                    lr = self.transform_IR_WOSCREENTONE(lr)
                if f == 1:
                    lr = TTF.center_crop(lr, (self.crop_height, self.crop_width))


            else:
                # 图像超分
                '''if random.uniform(0, 1) < 0.5:
                    f = 1
                    lr = self.transform_pad(lr)
                lr = self.transform_LR_TEST(lr)
                if f == 1:
                    lr = TT.functional.center_crop(lr, (self.crop_height, self.crop_width))
                lr = self.down_scale(lr)'''
                lr = TF.avg_pool2d(lr,2)
                #lr = resize(lr,64,64,"point")

        else:
            x = self.images[idx]
            x = self.rand_crop(x, self.crop_height, self.crop_width)
            x = x.float() / 255.
            x = self.transform_GT_VALID(x)
            lr = x.clone()
            hr = x.clone()

            if self.scale <= 1:
                lr = self.transform_IR_VALID(lr)
            else:
                lr = TF.avg_pool2d(lr, 2)

        return lr, hr

    def __len__(self):
        return len(self.images)


def load_data(batch_size, crop_height, crop_width, scale):
    num_workers = 8
    train_iter = torch.utils.data.DataLoader(
        TrainDataset(True, crop_height, crop_width, scale), batch_size,
        shuffle=True, drop_last=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(
        TrainDataset(False, crop_height, crop_width, scale), batch_size,
        drop_last=True, num_workers=num_workers)
    return train_iter, test_iter


if __name__ == "__main__":
    img = test_images("__SCREENTONE_14.png")
    #img = gaussian_noise(img,25/255,)
    img = jpeg_compression(img,25,False)
    img = TT.ToPILImage()(img)
    img.save("out2.png")

#triangle
