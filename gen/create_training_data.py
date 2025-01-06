import os
import sys
import argparse
from os import path
from tqdm import tqdm
import random
import torchvision.transforms.functional as TF
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision


def read_images(input_dir):
    mode = torchvision.io.image.ImageReadMode.RGB
    image_list = os.listdir(input_dir)
    ret_images = []
    for image in image_list:
        if image.split(".").pop() in ["png"]:
            img = torchvision.io.read_image(os.path.join(input_dir, image), mode)
            ret_images.append(img)



    return ret_images


def split_image(filepath_prefix, im, size, stride, reject_rate):
    c,h, w = im.shape
    rects = []
    for y in range(0, h, stride):
        if not y + size <= h:
            break
        for x in range(0, w, stride):
            if not x + size <= w:
                break
            rect = TF.crop(im, y, x, size, size)
            center = TF.center_crop(rect, (size // 2, size // 2))
            color_stdv = center.std(dim=[1, 2]).sum().item()
            rects.append((rect, color_stdv))

    n_reject = int(len(rects) * reject_rate)
    rects = [v[0] for v in sorted(rects, key=lambda v: v[1], reverse=True)][0:len(rects) - n_reject]

    index = 0
    for rect in rects:
        rect = TF.to_pil_image(rect)
        rect.save(f"{filepath_prefix}_{index}.png")
        index += 1


class CreateTrainingData(Dataset):
    def __init__(self,input_dir,output_dir,prefix,size,stride,reject_rate):
        super().__init__()
        self.files = read_images(input_dir)
        self.filename_prefix = prefix + "_"
        self.output_dir = output_dir
        self.size = size
        self.stride = stride
        self.reject_rate = reject_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        file = self.files[i]
        file = file.float() / 255.
        stride = int(self.size * self.stride)

        split_image(path.join(self.output_dir, self.filename_prefix + str(i)),file, self.size, stride, self.reject_rate)

        return 0


def main():
    num_workers = 8

    input_dir = r"E:\Encode\Dataset\Raw_Data\YAPD"
    image_type = "process"
    if not os.path.exists(os.path.join(input_dir,"output")):
        os.makedirs(os.path.join(input_dir,"output"), exist_ok=True)

    loader = DataLoader(
        CreateTrainingData(os.path.join(input_dir,image_type), os.path.join(input_dir,"output"),image_type,size=256,stride=0.5,reject_rate=0.5),
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=4,
        drop_last=False
    )
    for _ in tqdm(loader, ncols=80):
        pass

if __name__ == "__main__":
    main()