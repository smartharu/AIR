import math
import numpy as np
import torch
from utils import logger
from torch.optim.adamw import AdamW
import os
import torch.nn as nn
from nets import sunet_arch, srvgg_arch, span_arch
from data import dataset, dataset_neo
import traceback


class Trainer:
    def __init__(self):

        self.scale = 2
        self.batch_size = 64
        self.crop_size = 128
        self.device = torch.device("cuda:0")
        # self.netG = sunet_arch.UNetResA(3, 3, [32, 64, 96, 128], [4, 4, 4, 4], True).to(self.device)
        # self.netG = srvgg_arch.SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=24, num_conv=8, upscale=2).to(self.device)
        self.netG = span_arch.SPAN(3, 3, 48, 2, True).to(self.device)
        # self.netG = sunet_arch.SPAU(bias=True).to(self.device)
        # self.netG = srvgg_arch.SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=48, num_conv=16, upscale=2, act_type='prelu').to(self.device)

        self.model_name = self.netG.get_model_name()

        self.epochs = 1000
        self.cur_epoch = 0
        self.restart_epoch = False
        self.lr = 2e-4
        # self.trainerG = Adam(self.netG.parameters(), lr=self.lr)
        self.trainerG = AdamW(self.netG.parameters(), lr=self.lr, weight_decay=0.001)

        self.load_model()
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.trainerG, milestones=[100, 200, 300, 400], gamma=0.5,
                                                              last_epoch=self.cur_epoch - 1)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.trainerG,T_0=10,eta_min=self.lr * 5e-3)
        # self.update_lr()
        self.train_iter, self.test_iter = dataset_neo.load_data(self.batch_size, self.crop_size, self.crop_size,
                                                                scale=self.scale)
        self.total_step = len(self.train_iter)

        self.content_loss_factor = 1.0

    def train(self):
        content_criterion = nn.L1Loss().to(self.device)
        self.netG.train()
        logger.info(f"network:{self.model_name}, scale:{self.scale}")
        logger.info("Start training!")

        for epoch in range(self.cur_epoch + 1, self.epochs + 1):
            logger.info(f"Epoch:{epoch}")
            psnr = []
            for iteration, data in enumerate(self.train_iter):
                blur, sharp = data
                blur = blur.to(self.device)
                sharp = sharp.to(self.device)

                self.trainerG.zero_grad()
                fake_sharp = self.netG(blur)

                content_loss = content_criterion(fake_sharp, sharp)
                temp_psnr = cal_psnr(fake_sharp, sharp)

                if epoch > 1 and temp_psnr < 0:
                    logger.info("psnr error! Try to restart training.")
                    raise ValueError("psnr error! try to restart training.")
                psnr.append(temp_psnr)

                generator_loss = content_loss * self.content_loss_factor

                generator_loss.backward()
                self.trainerG.step()

                if iteration % 80 == 0:
                    logger.info(
                        f"[TRAIN_PSNR {temp_psnr:.4f}] [LR {self.trainerG.state_dict()['param_groups'][0]['lr']}]")

            self.scheduler.step()
            train_psnr = np.mean(psnr)
            val_psnr = evaluate_accuracy(self.netG, self.test_iter, self.device)

            torch.save({'model_state_dict': self.netG.state_dict(), 'trainer_state_dict': self.trainerG.state_dict(),
                        'epoch': epoch}, self.model_name + ".pth")
            if epoch % 20 == 0:
                torch.save({'model_state_dict': self.netG.state_dict(),
                            'trainer_state_dict': self.trainerG.state_dict(),
                            'epoch': epoch}, self.model_name + "_epoch_" + str(epoch) + ".pth")
            logger.info(
                f"[EPOCH {epoch}] [TRAIN_PSNR {train_psnr:.4f}] [VAL_PSNR {val_psnr:.4f}] [LR {self.trainerG.state_dict()['param_groups'][0]['lr']}]")

    def load_model(self):
        if os.path.exists(self.model_name + ".pth"):
            logger.info(f"{self.model_name}加载模型")
            checkpointG = torch.load(self.model_name + ".pth")
            self.netG.load_state_dict(checkpointG['model_state_dict'])
            self.trainerG.load_state_dict(checkpointG['trainer_state_dict'])
            self.cur_epoch = checkpointG['epoch']

        if self.restart_epoch:
            self.cur_epoch = 0

        logger.info(self.trainerG.state_dict()['param_groups'][0]['lr'])

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

    np.seterr(invalid='ignore')
    cnt = 50
    while cnt:
        cnt -= 1
        try:
            t = Trainer()
            t.train()
        except:
            print(traceback.format_exc())
            continue
        else:
            break

    # os.system("shutdown -s")
