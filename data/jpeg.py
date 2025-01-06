import math

import torch
import cv2
import numpy as np
import random
from torchvision.transforms import functional as TTF
from data.datatools import load_image
from typing import Any, Mapping, Optional, Sequence, Union, Tuple
from data.noise import gaussian_noise

LAPLACIAN_KERNEL = torch.tensor([
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0],
], dtype=torch.float32).reshape(1, 1, 3, 3)


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


def choose_jpeg_quality(style: str, noise_level: int) -> Sequence[int]:
    qualities = []
    if style == "art":
        if noise_level == 0:
            qualities.append(random.randint(85, 95))
        elif noise_level == 1:
            qualities.append(random.randint(65, 85))
        elif noise_level in {2, 3}:
            # 2 and 3 are the same, NR_RATE is different
            r = random.uniform(0, 1)
            if r > 0.4:
                qualities.append(random.randint(27, 70))
            elif r > 0.1:
                # nunif: Add high quality patterns
                if random.uniform(0, 1) < 0.05:
                    quality1 = random.randint(37, 95)
                else:
                    quality1 = random.randint(37, 70)
                quality2 = quality1 - random.randint(5, 10)
                qualities.append(quality1)
                qualities.append(quality2)
            else:
                # nunif: Add high quality patterns
                if random.uniform(0, 1) < 0.05:
                    quality1 = random.randint(52, 95)
                else:
                    quality1 = random.randint(52, 70)
                quality2 = quality1 - random.randint(5, 15)
                quality3 = quality1 - random.randint(15, 25)
                qualities.append(quality1)
                qualities.append(quality2)
                qualities.append(quality3)
    elif style == "photo":
        if noise_level == 0:
            qualities.append(random.randint(85, 95))
        elif noise_level == 1:
            if random.uniform(0, 1) < 0.5:
                qualities.append(random.randint(37, 70))
            else:
                qualities.append(random.randint(90, 98))
        else:
            if noise_level == 3 or random.uniform(0, 1) < 0.6:
                if random.uniform(0, 1) < 0.05:
                    quality1 = random.randint(52, 95)
                else:
                    quality1 = random.randint(37, 70)
                qualities.append(quality1)
                if quality1 >= 70 and random.uniform(0, 1) < 0.2:
                    qualities.append(random.randint(70, 90))
            else:
                qualities.append(random.randint(90, 98))
    else:
        raise NotImplementedError()

    return qualities

def anime_jpeg_quality() -> Sequence[int]:
    qualities = []
    if random.uniform(0,1)<0.4:
        qualities.append(random.randint(75, 95))
    if random.uniform(0,1)<0.4:
        qualities.append(random.randint(25, 75))
    return qualities

def sharpen(x: torch.Tensor, strength: float = 0.1) -> torch.Tensor:
    grad = torch.nn.functional.conv2d(x.mean(dim=0, keepdim=True).unsqueeze(0),
                                      weight=LAPLACIAN_KERNEL, stride=1, padding=1).squeeze(0)
    x = x + grad * strength
    #x = torch.clamp(x, 0., 1.)
    return x


def sharpen_noise(original_x: torch.Tensor, noise_x: torch.Tensor, strength: float = 0.1) -> torch.Tensor:
    """ shapen (noise added image - original image) diff
    """
    noise = noise_x - original_x
    noise = sharpen(noise, strength=strength)
    x = torch.clamp(original_x + noise, 0., 1.)
    return x


def sharpen_noise_all(x: torch.Tensor, strength: float = 0.1) -> torch.Tensor:
    """ just sharpen image
    """
    x = sharpen(x, strength=strength)
    return x

def shift_jpeg_block(x:torch.Tensor,scale:int=2,x_shift=None):
    # nunif: Add random crop before the second jpeg
    c,h,w = x.shape
    if x_shift is None:
        if random.uniform(0, 0.5) < 0.5:
            h_shift = random.randint(0, 7)
            w_shift = random.randint(0, 7)
        else:
            h_shift = w_shift = 0
    else:
        h_shift = w_shift = x_shift

    if h_shift > 0 or w_shift > 0:
        h_shift = h_shift * scale
        w_shift = w_shift * scale
        x = TTF.crop(x, h_shift, w_shift, h - h_shift, w - w_shift)
        #y = TTF.crop(y, h_shift, w_shift, y_h - y_h_shift, y_w - y_w_shift)
        #assert y.size[0] == x.size[0] * y_scale and y.size[1] == x.size[1] * y_scale

    return x

class RandomJPEGNoise:
    def __init__(self, prob: float = 0.6, jpeg_q: int | Tuple[int, int] = 95,
                 css_prob: float = 0.5,sharpen_prob:float=0.2) -> None:
        self.prob = prob
        self.css_prob = css_prob
        self.jpeg_q = jpeg_q
        self.sharpen_prob = sharpen_prob

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

        x_noise = jpeg_compression(x, jpeg_qualities, chroma_subsampling)

        if random.uniform(0, 1) < self.sharpen_prob:
            x_noise = sharpen_noise(x, x_noise, strength=random.uniform(0.05, 0.2))
        return x_noise

class RandomAnimeNoise:
    def __init__(self, prob:float = 1.0,gaussian_prob:float = 0.2) -> None:
        self.prob = prob
        self.gaussian_prob = gaussian_prob

    def __call__(self, x:torch.Tensor) -> torch.Tensor:
        if random.uniform(0, 1) > self.prob:
            return x
        qualities = anime_jpeg_quality()
        if len(qualities) == 0:
            return x

        if random.uniform(0,1) < self.gaussian_prob:
            min_quality = min(qualities)

            if random.uniform(0, 1) < 0.2:
                blur = True
            else:
                blur = False

            if random.uniform(0, 1) < 0.5:
                gray = True
            else:
                gray = False

            if min_quality <= 75:
                gaussian_factor = random.randint(0, 15)
            else:
                gaussian_factor = math.floor(60 - min_quality * 0.6)

            x = gaussian_noise(x,gaussian_factor / 255.,gray,blur)
        origin_x = x
        for q in qualities:
            x = jpeg_compression(x, q, False)

        if random.uniform(0, 1) < 0.2:
            x = sharpen_noise(origin_x, x, strength=random.uniform(0.05, 0.2))
        return x


class RandomBlock:
    def __init__(self, prob: float = 1.0, noise_level: int = 3,
                 css_prob: float = 0.5) -> None:
        self.prob = prob
        self.css_prob = css_prob
        self.noise_level = noise_level
        self.style = "art"

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        original_x = x
        if random.uniform(0, 1) > self.prob:
            # use lower noise_level noise
            # this is the fix for a problem in the original waifu2x
            # that lower level noise cannot be denoised with higher level denoise model.
            min_level = -1 if self.noise_level < 2 else 0
            noise_level = random.randint(min_level, self.noise_level - 1)
            if noise_level == -1:
                # do nothing
                return x
        else:
            # use noise level noise
            noise_level = self.noise_level

        qualities = choose_jpeg_quality(self.style, noise_level)
        assert len(qualities) > 0

        if random.uniform(0, 1) < self.css_prob:
            subsampling = True
        else:
            subsampling = False

        for i, quality in enumerate(qualities):
            x = jpeg_compression(x, quality, subsampling)
            if i == 0 and self.style == "photo" and noise_level in {2, 3} and random.uniform(0, 1) < 0.2:
                if random.uniform(0, 1) < 0.75:
                    x = sharpen_noise(original_x, x, strength=random.uniform(0.05, 0.2))
                else:
                    x = sharpen_noise_all(x, strength=random.uniform(0.1, 0.3))
        return x

if __name__ == "__main__":
    img = load_image("lun39b.png")
    img = RandomAnimeNoise(prob=1.0,gaussian_prob=0.2)(img)
    #res = sharpen_noise(original_x=img, noise_x=img_noise, strength= 0.2)
    res = TTF.to_pil_image(img)
    res.save("jpeg.png")