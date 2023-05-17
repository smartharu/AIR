import math
import numpy as np
import torch
from torch.optim.adam import Adam
import os
import torch.nn as nn
from nets import sunet
from data import dataset


class Trainer:
    def __init__(self):

        self.scale = 2

        self.batch_size = 16
        self.crop_size = 128
        self.device = torch.device("cuda:0")
        self.netG = sunet.SUNet(scale=self.scale).to(self.device)

        self.model_name = self.netG.get_model_name()

        self.epochs = 2000
        self.cur_epoch = 0
        self.lr = 0.0002
        self.trainerG = Adam(self.netG.parameters(), lr=self.lr)

        self.load_model()
        self.update_lr()
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.trainerG, milestones=[1000], gamma=0.5,last_epoch=self.cur_epoch-1)
        self.train_iter, self.test_iter = dataset.load_data(self.batch_size, self.crop_size, self.crop_size,
                                                            scale=self.scale)
        self.total_step = len(self.train_iter)
        self.step_index = 0

        self.content_loss_factor = 1.0

    def train(self):

        content_criterion = nn.L1Loss().to(self.device)
        self.netG.train()
        for epoch in range(self.cur_epoch+1, self.cur_epoch+self.epochs + 1):
            self.step_index = 0
            psnr = []

            for blur, sharp in self.train_iter:
                blur = blur.to(self.device)
                sharp = sharp.to(self.device)

                self.trainerG.zero_grad()
                fake_sharp = self.netG(blur)

                content_loss = content_criterion(fake_sharp, sharp)
                psnr.append(cal_psnr(fake_sharp, sharp))

                generator_loss = content_loss * self.content_loss_factor

                generator_loss.backward()
                self.trainerG.step()

                self.step_index += 1

                '''
                if self.step_index % 64 == 0:
                    print(f"[Epoch {epoch}]"
                          f"[G loss {generator_loss.item():.4f}]"
                          f"[lr {self.trainerG.state_dict()['param_groups'][0]['lr']}]"
                          f"[progress {self.step_index / self.total_step * 100:.2f}%]"
                          f"[PSNR {psnr:.4f}]")
                '''
            self.scheduler.step()
            train_psnr = np.mean(psnr)
            val_psnr = evaluate_accuracy(self.netG, self.test_iter, self.device)

            torch.save({'model_state_dict': self.netG.state_dict(),
                        'trainer_state_dict': self.trainerG.state_dict(),
                        'epoch': epoch}, self.model_name+".pth")
            print(f"[EPOCH {epoch}]"
                  f"[TRAIN_PSNR {train_psnr:.4f}]"
                  f"[VAL_PSNR {val_psnr:.4f}]")

    def load_model(self):
        if os.path.exists(self.model_name+".pth"):
            print(f"{self.model_name}加载模型")
            checkpointG = torch.load(self.model_name+".pth")
            self.netG.load_state_dict(checkpointG['model_state_dict'])
            self.trainerG.load_state_dict(checkpointG['trainer_state_dict'])
            self.cur_epoch = checkpointG['epoch']

    def update_lr(self):
        for p in self.trainerG.param_groups:
            p['lr'] = self.lr


def cal_psnr(img1, img2):
    n = img1.numel()
    mse = math.fabs(torch.sum((img1 - img2) ** 2) / n)
    # print("mse",mse)
    if mse > 0:
        psnr = 20 * math.log10(1.0 / math.sqrt(mse))
    else:
        psnr = 0
    return psnr


def evaluate_accuracy(net, data_iter, device):
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    ret = []
    with torch.no_grad():
        for val_blur, val_sharp in data_iter:
            val_blur = val_blur.to(device)
            val_sharp = val_sharp.to(device)
            ret.append(cal_psnr(net(val_blur), val_sharp))
    return max(ret)


if __name__ == '__main__':
    t = Trainer()
    t.train()
