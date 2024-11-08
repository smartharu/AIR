import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as TT
import torchvision.transforms.functional as TTF
from torch.nn import functional as F
from torch.utils import data
import os
from .jpeg import RandomJPEGNoise, RandomBlock
from .transform import RandomBlur, RandomGray, RandomGaussianBlur, RandomLanczosFilter, RandomSincFilter, RandomSharpen
from .rescale import RandomRescale, A4KRandomDownscale, AntialiasX
from .noise import RandomNoise
from .utils import RamdomAugBorder
from utils.logger import logger
from typing import Sequence, Tuple

np.seterr(invalid='ignore')


def read_images(is_train: bool = True) -> Tuple[Sequence[torch.Tensor], Sequence[str]]:
    mode = torchvision.io.image.ImageReadMode.RGB
    labels = []
    filenames = []
    #dataset = "EAIRDMS_PLUS"
    dataset = r"E:\Encode\Dataset\EAIRDMS_PLUS"
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


def load_images(dir: str) -> torch.Tensor:
    mode = torchvision.io.image.ImageReadMode.RGB
    image = torchvision.io.read_image(dir, mode)
    return image.float() / 255.


def rand_crop(img, height, width):
    rect = torchvision.transforms.RandomCrop.get_params(
        img, (height, width))
    img = torchvision.transforms.functional.crop(img, *rect)
    return img


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, is_train, crop_height, crop_width, scale):
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.is_train = is_train
        self.scale = scale

        images, names = read_images(is_train)  # images list

        self.image_counts = len(images)
        self.images = images

        self.transform_GT = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomInvert(0.5),
                TT.RandomApply([TT.ColorJitter(brightness=(0, 2), contrast=(0, 2), saturation=(0, 2), hue=(-0.5, 0.5))],
                               p=0.5),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.RandomVerticalFlip(p=0.5),
                # RandomRotate(prob=0.5),
                RandomBlur(prob=0.3, kernel_size=(9, 23), sigma=(2.5, 5.0), radius=(2, 5)),
                RandomGray(prob=0.3)
            ]
        )

        self.transform_GT_VALID = torchvision.transforms.Compose(
            [
                RandomGaussianBlur(prob=0.1, kernel_size=(9, 23), sigma=(3.0, 5.0))
            ]
        )

        self.transform_IDENTITY = TT.Compose(
            [
                RandomGaussianBlur(prob=0.4, kernel_size=(3, 21), sigma=(0.2, 0.5)),
            ]
        )


        self.transform_IR_TEST = TT.Compose(
            [

                TT.RandomOrder([
                    TT.RandomChoice([
                        RandomLanczosFilter(prob=0.4, kernel_size=(7, 21), sigma=(2.0, 5.0)),
                        RandomSincFilter(prob=0.4, kernel_size=(3, 21), sigma=(2.0, np.pi)),
                    ]),
                    RandomSharpen(prob=0.4, gray_prob=0.5, sigma=(0.2, 2.0)),
                    RandomRescale(prob=0.8, task="first", anisotropic_p=.3),
                    RandomGaussianBlur(prob=0.4, kernel_size=(3, 21), sigma=(0.2, 0.5)),
                    TT.RandomChoice([
                        RandomRescale(prob=0.4, task="second", anisotropic_p=.3),
                        RandomBlur(prob=0.4, kernel_size=(3, 21), sigma=(0.1, 1.0), radius=1)
                    ]),
                    RandomJPEGNoise(prob=0.4, jpeg_q=(95, 100), css_prob=1.0)
                ]),
                TT.RandomOrder([
                    RandomNoise(prob=0.4, gaussian_factor=25, gray_prob=0.5, blur_prob=0.2),
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

        self.transform_IR_TEST_V = TT.Compose(
            [
                RandomRescale(prob=1.0, task="first", anisotropic_p=0.3),
                RandomNoise(prob=1.0, gaussian_factor=25, gray_prob=0.5, blur_prob=0.3),
                RandomJPEGNoise(prob=1.0, jpeg_q=(25, 95), css_prob=0.0)
            ]
        )

        self.transform_LR_TEST = torchvision.transforms.Compose(
            [
                TT.RandomOrder([
                    A4KRandomDownscale(scale_factor=self.scale),
                    RandomGaussianBlur(prob=0.1, kernel_size=(3, 21), sigma=(0.2, 0.5)),
                    TT.RandomChoice([
                        RandomLanczosFilter(prob=0.1, kernel_size=(7, 21), sigma=(2.0, 5.0)),
                        RandomSincFilter(prob=0.1, kernel_size=(3, 21), sigma=(2.0, np.pi)),
                    ]),
                    RandomJPEGNoise(prob=0.5, jpeg_q=(95, 100), css_prob=1.0)
                ])

            ]
        )

        self.transform_pad = TT.Pad(padding=8)

        self.transform_LR_TEST_V = torchvision.transforms.Compose(
            [
                TT.RandomOrder([
                    A4KRandomDownscale(scale_factor=self.scale),
                    TT.RandomChoice([
                        RandomLanczosFilter(prob=0.1, kernel_size=(7, 21), sigma=(2.0, 5.0)),
                        RandomSincFilter(prob=0.1, kernel_size=(3, 21), sigma=(2.0, np.pi)),
                    ]),
                    RandomGaussianBlur(prob=0.1, kernel_size=(3, 21), sigma=(0.2, 0.5)),
                    RandomJPEGNoise(prob=0.5, jpeg_q=(95, 100), css_prob=1.0)
                ])
            ]
        )

        logger.info('read ' + str(self.image_counts) + ' pictures')
        logger.info('read ' + str(len(self.images)) + ' examples')

    def __getitem__(self, idx):
        if self.is_train:
            f = 0
            x = self.images[idx]
            x = rand_crop(x, self.crop_height, self.crop_width)
            x = x.float() / 255.

            x = self.transform_GT(x)

            lr = x.clone()
            hr = x.clone()

            if self.scale <= 1:
                # 图像恢复
                if random.uniform(0, 1) < 0.5:
                    f = 1
                    lr = self.transform_pad(lr)
                    # print(lr.shape)

                lr = self.transform_IR_TEST(lr)
                lr = self.transform_IDENTITY(lr)
                # lr = self.transform_EDGE(lr)

                if f == 1:
                    lr = TT.functional.center_crop(lr, [self.crop_height, self.crop_width])


            else:
                # 图像超分
                lr = self.transform_LR_TEST(lr)
                #lr = TTF.resize(lr,size = [lr.shape[2]//2,lr.shape[1]//2],interpolation=TT.InterpolationMode.NEAREST)
                #lr = F.avg_pool2d(lr,self.scale)
        else:
            x = self.images[idx]
            x = rand_crop(x, self.crop_height, self.crop_width)
            x = x.float() / 255.
            x = self.transform_GT_VALID(x)

            lr = x.clone()
            hr = x.clone()

            if self.scale <= 1:
                lr = self.transform_IR_TEST_V(lr)
            else:
                lr = self.transform_LR_TEST_V(lr)

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
    x = torch.randn((3,128,128))

    for i in range(100):

        res = RandomRescale(prob=1.0, task="first", anisotropic_p=.3)(x)
