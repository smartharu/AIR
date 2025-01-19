import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as TT
import torchvision.transforms.functional as TTF
from torch.nn import functional as F
from torch.utils import data
import os
from data.jpeg import RandomJPEGNoise, RandomAnimeNoise, RandomBlock
from data.transform import RandomBlur, RandomGray, RandomGaussianBlur, RandomLanczosFilter, RandomSincFilter, RandomSharpen,RandomSafeRotate
from data.rescale import RandomRescale,RandomDownscale
from data.noise import RandomNoise
from data.datatools import RamdomAugBorder,load_image
from utils.logger import logger
from typing import Sequence, Tuple

np.seterr(invalid='ignore')


def read_images(is_train: bool = True, dataset_dir:str = None) -> Tuple[Sequence[torch.Tensor], Sequence[str]]:
    mode = torchvision.io.image.ImageReadMode.RGB
    labels :list[torch.Tensor]= []
    filenames:list[str] = []
    #dataset = r"E:\Encode\Dataset\EAIRDMS_PLUS"
    if is_train:
        train_list = os.listdir(os.path.join(dataset_dir, "train"))
        for train_img in train_list:
            if train_img.split(".").pop() in ["png"]:
                img = torchvision.io.read_image(os.path.join(dataset_dir, "train", train_img), mode)
                labels.append(img)
                filenames.append(train_img)
    else:
        valid_list = os.listdir(os.path.join(dataset_dir, "valid"))
        for valid_img in valid_list:
            if valid_img.split(".").pop() in ["png"]:
                img = torchvision.io.read_image(os.path.join(dataset_dir, "valid", valid_img), mode)
                labels.append(img)
                filenames.append(valid_img)
    return labels, filenames

def rand_crop(img, height, width):
    rect = torchvision.transforms.RandomCrop.get_params(
        img, (height, width))
    img = torchvision.transforms.functional.crop(img, *rect)
    return img


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, is_train,crop_height, crop_width, scale,dataset_dir):
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.is_train = is_train
        self.scale = scale

        self.images, self.names, = read_images(is_train,dataset_dir)  # images list

        self.image_counts = len(self.images)

        self.transform_GT = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomInvert(0.5),
                TT.RandomApply([TT.ColorJitter(brightness=(0, 2), contrast=(0, 2), saturation=(0, 2), hue=(-0.5, 0.5))],
                               p=0.5),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.RandomVerticalFlip(p=0.5),
                #RandomRotate(prob=0.5),
                RandomBlur(prob=0.3, kernel_size=(9, 23), sigma=(2.5, 5.0), radius=(2, 5)),
                RandomGray(prob=0.3)
            ]
        )

        self.transform_ROATE = RandomSafeRotate(prob=0.3)

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
                    RandomGaussianBlur(prob=0.4, kernel_size=(3, 21), sigma=(0.2, 0.6)),
                    TT.RandomChoice([
                        RandomRescale(prob=0.4, task="second", anisotropic_p=.3),
                        RandomBlur(prob=0.4, kernel_size=(3, 21), sigma=(0.1, 1.0), radius=1)
                    ]),
                    RandomJPEGNoise(prob=0.4, jpeg_q=(95, 100), css_prob=1.0)
                ]),
                TT.RandomOrder([
                    RandomNoise(prob=0.2, gaussian_factor=25, gray_prob=0.5, blur_prob=0.2),
                    RandomAnimeNoise(prob=0.6, gaussian_prob=0.0)
                ])
            ]
        )

        self.transform_LR_TEST = TT.Compose(
            [

                TT.RandomOrder([
                    TT.RandomChoice([
                        RandomLanczosFilter(prob=0.4, kernel_size=(7, 21), sigma=(2.0, 5.0)),
                        RandomSincFilter(prob=0.4, kernel_size=(3, 21), sigma=(2.0, np.pi)),
                    ]),
                    RandomSharpen(prob=0.4, gray_prob=0.5, sigma=(0.2, 2.0)),
                    RandomGaussianBlur(prob=0.8, kernel_size=(3, 21), sigma=(0.2, 0.6)),
                    TT.RandomChoice([
                        RandomRescale(prob=0.7, task="second", anisotropic_p=.3),
                        RandomBlur(prob=0.7, kernel_size=(3, 21), sigma=(0.1, 1.0), radius=1)
                    ]),
                    RandomJPEGNoise(prob=0.4, jpeg_q=(95, 100), css_prob=1.0)
                ]),
                RandomDownscale(scale_factor=self.scale),
                RandomJPEGNoise(prob=0.6, jpeg_q=(25, 95), css_prob=0.0)
            ]
        )

        self.transform_IR_SCREENTONE_TEST = TT.Compose(
            [

                TT.RandomOrder([
                    TT.RandomChoice([
                        RandomLanczosFilter(prob=0.4, kernel_size=(7, 21), sigma=(2.0, 5.0)),
                        RandomSincFilter(prob=0.4, kernel_size=(3, 21), sigma=(2.0, np.pi)),
                    ]),
                    RandomSharpen(prob=0.4, gray_prob=0.5, sigma=(0.2, 2.0)),
                    RandomRescale(prob=0.8, task="second", anisotropic_p=.3),
                    RandomGaussianBlur(prob=0.4, kernel_size=(3, 21), sigma=(0.2, 0.6)),
                    TT.RandomChoice([
                        RandomRescale(prob=0.4, task="second", anisotropic_p=.3),
                        RandomBlur(prob=0.4, kernel_size=(3, 21), sigma=(0.1, 1.0), radius=1)
                    ]),
                    RandomJPEGNoise(prob=0.4, jpeg_q=(95, 100), css_prob=1.0)
                ]),
                TT.RandomOrder([
                    RandomNoise(prob=0.2, gaussian_factor=25, gray_prob=0.5, blur_prob=0.2),
                    RandomAnimeNoise(prob=0.6, gaussian_prob=0.0)
                ])
            ]
        )

        self.transform_DEBLOCK = TT.Compose(
            [
                TT.RandomOrder([
                    RandomBlock(prob=1.0),
                    RandomGaussianBlur(prob=0.1, kernel_size=(3, 21), sigma=(0.2, 0.5)),
                    RandomLanczosFilter(prob=0.1, kernel_size=(3, 5), sigma=(2.0, 5.0)),
                ])

            ]
        )

        self.transform_DEBLOCK_V = TT.Compose(
            [
                TT.RandomOrder([
                    RandomJPEGNoise(prob=1.0, jpeg_q=50, css_prob=0.5)
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

        self.transform_pad = TT.Pad(padding=8)

        logger.info('read ' + str(self.image_counts) + ' pictures')
        logger.info('read ' + str(len(self.images)) + ' examples')

    def __getitem__(self, idx):
        if self.is_train:
            f = 0
            x = self.images[idx]
            x = rand_crop(x, self.crop_height, self.crop_width)
            x = x.float() / 255.
            if "LINEPATTERN" in self.names:
                x = self.transform_ROATE(x)
            x = self.transform_GT(x)

            lr = x.clone()
            hr = x.clone()

            if self.scale <= 1:
                # 图像恢复
                if random.uniform(0, 1) < 0.5:
                    f = 1
                    lr = self.transform_pad(lr)
                    # print(lr.shape)
                if "LINEPATTERN" in self.names:
                    lr = self.transform_IR_SCREENTONE_TEST(lr)
                else:
                    lr = self.transform_IR_TEST(lr)
                #lr = self.transform_DEBLOCK(lr)
                if f == 1:
                    lr = TT.functional.center_crop(lr, [self.crop_height, self.crop_width])
            else:
                # 图像超分
                #lr = self.transform_LR_TEST(lr)
                #lr = TTF.resize(lr,size = [lr.shape[2]//2,lr.shape[1]//2],interpolation=TT.InterpolationMode.NEAREST)
                lr = self.transform_LR_TEST(lr)
        else:
            x = self.images[idx]
            x = rand_crop(x, self.crop_height, self.crop_width)
            x = x.float() / 255.
            x = self.transform_GT_VALID(x)

            lr = x.clone()
            hr = x.clone()

            if self.scale <= 1:
                lr = self.transform_IR_TEST_V(lr)
                #lr = self.transform_DEBLOCK_V(lr)
            else:
                lr = F.avg_pool2d(lr,self.scale)

        return lr, hr

    def __len__(self) -> int:
        return len(self.images)


def load_data(batch_size:int, crop_height:int, crop_width:int, scale:int,dataset_dir:str):
    num_workers = 8
    train_iter = torch.utils.data.DataLoader(
        TrainDataset(True, crop_height, crop_width, scale,dataset_dir), batch_size,
        shuffle=True, drop_last=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(
        TrainDataset(False, crop_height, crop_width, scale,dataset_dir), batch_size,
        drop_last=True, num_workers=num_workers)
    return train_iter, test_iter



if __name__ == "__main__":
    print("Test")
