import random
import math
import torch
from data.color import rgb_to_yuv, yuv_to_rgb
from data.rescale import rescale
from data.datatools import get_platform
from data.datatools import load_image
from torchvision.transforms import functional as TTF
import numpy as np

def gaussian_noise(img: torch.Tensor, sigma: float, gray_noise: bool = False,
                   blur_noise: bool = False) -> torch.Tensor:
    source_width = img.shape[2]

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

    platform = get_platform()

    min_length = math.floor(source_width*0.375)
    max_length = math.floor(source_width*0.5)
    anisotropic_length = math.floor(source_width*0.25)
    if blur_noise:
        if platform == "win":
            rg = rescale(rg, min_length, max_length,0,anisotropic_length)
        else:
            rg = rg + 0.5
            rg = rescale(rg, min_length, max_length,0,anisotropic_length)
            rg = rg - 0.5
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



def poisson_noise(img: torch.Tensor, scale: float, gray_noise: bool = False,blur_noise:bool=False):
    source_width = img.shape[2]
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
    noise = torch.from_numpy(noise)

    platform = get_platform()
    min_length = math.floor(source_width * 0.375)
    max_length = math.floor(source_width * 0.5)
    anisotropic_length = math.floor(source_width * 0.25)

    if blur_noise:
        if platform == "win":
            noise = rescale(noise, min_length, max_length,0,anisotropic_length)
        else:
            noise = noise + 0.5
            noise = rescale(noise, min_length, max_length,0,anisotropic_length)
            noise = noise - 0.5
    else:
        pass

    out = (img + noise).clamp(0,1)

    if gray_noise:
        add = rgb_to_yuv(out)
        y_add = add[..., 0, :, :]
        u_add = add[..., 1, :, :]
        v_add = add[..., 2, :, :]
        res = yuv_to_rgb(torch.stack([y_add, u, v], -3))
    else:
        res = out

    return res.clamp(0, 1)




class RandomNoise:
    def __init__(self, prob: float = 0.65, gaussian_factor: int = 25, poisson_scale:float=2.5,gray_prob: float = 0.5,
                 blur_prob: float = 0.3) -> None:
        self.prob = prob
        self.gaussian_factor = gaussian_factor
        self.poisson_scale = poisson_scale
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

        if random.uniform(0,1)<0.5:
            x = gaussian_noise(x, sigma / 255., gray_noise, blur_noise)
        else:
            x = poisson_noise(x,random.uniform(0.02,self.poisson_scale),gray_noise,blur_noise)

        return x.clamp(0, 1)

if __name__ == "__main__":
    img = load_image("lun39b.png")
    img = poisson_noise(img, scale=2.5, gray_noise = True,blur_noise=True)
    #img = jpeg_compression(img,95,True)
    #res = sharpen_noise(original_x=img, noise_x=img_noise, strength= 0.2)
    res = TTF.to_pil_image(img)
    res.save("jpeg.png")