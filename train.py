import math
import numpy as np
import torch
from utils import logger,set_log_name
import utils
from torch.optim.adamw import AdamW
import os
import torch.nn as nn
from nets import SPAN,UNetResA,SRVGGNetCompact,UpUNetResA
from data import dataset
import random
import traceback

class Trainer:
    def __init__(self):

        self.scale = 1
        self.batch_size = 32
        self.crop_size = 128
        self.dataset = r"E:\Encode\Dataset\YAPD"
        self.device = torch.device("cuda:0")

        self.netG = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=8, num_conv=8, upscale=1).to(self.device)
        #self.netG = UpUNetResA(3,3,2,[32,64,128,256],[4,4,4,4],True).to(self.device)

        self.model_name = self.netG.get_model_name()

        self.epochs = 1000
        self.cur_epoch = 0
        self.restart_epoch = False
        self.lr = 1e-4
        # self.trainerG = Adam(self.netG.parameters(), lr=self.lr)
        self.trainerG = AdamW(self.netG.parameters(), lr=self.lr, weight_decay=0.001)

        self.load_model()
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.trainerG, milestones=[200, 400, 600, 800], gamma=0.5,
                                                              last_epoch=self.cur_epoch - 1)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.trainerG,T_0=10,eta_min=self.lr * 5e-3)
        # self.update_lr()
        self.train_iter, self.test_iter = dataset.load_data(self.batch_size, self.crop_size, self.crop_size, self.scale,self.dataset)
        self.total_step = len(self.train_iter)

        self.content_loss_factor = 1.0
        set_log_name(self.model_name)
    def train(self):
        setup_seed()
        content_criterion = nn.L1Loss().to(self.device)
        self.netG.train()
        logger.info(f"NETWORK:{self.model_name}, SCALE:{self.scale}")
        logger.info("Start training!")

        for epoch in range(self.cur_epoch + 1, self.epochs + 1):
            logger.info(f"EPOCH:{epoch} LEARNING_RATE:{self.trainerG.state_dict()['param_groups'][0]['lr']}")
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
                        f"[TRAIN_PSNR {temp_psnr:.4f}] [LR {self.trainerG.state_dict()['param_groups'][0]['lr']}] [LOSS] {content_loss:.4f}")

            self.scheduler.step()
            self.cur_epoch += 1
            train_psnr = np.mean(psnr)
            val_psnr = evaluate_accuracy(self.netG, self.test_iter, self.device)

            self.save_model(epoch,False)
            if epoch % 20 == 0:
                self.save_model(epoch,True)
            logger.info(
                f"[EPOCH {epoch}] [TRAIN_PSNR {train_psnr:.4f}] [VAL_PSNR {val_psnr:.4f}] [LR {self.trainerG.state_dict()['param_groups'][0]['lr']}]")

    def load_model(self):
        if os.path.exists(self.model_name + ".pth"):
            logger.info(f"Load model: {self.model_name}")
            checkpointG = torch.load(self.model_name + ".pth")
            self.netG.load_state_dict(checkpointG['model_state_dict'])
            self.trainerG.load_state_dict(checkpointG['trainer_state_dict'])
            self.cur_epoch = checkpointG['epoch']
            logger.info(f"Start from epoch: {self.cur_epoch}, learning_rate: {self.trainerG.state_dict()['param_groups'][0]['lr']}")

        if self.restart_epoch:
            self.cur_epoch = 0
            logger.info("Restart epoch!")

    def save_model(self,epoch,check=False):
        if check:
            saved_model_name = f"{self.model_name}_epoch{epoch}.pth"
        else:
            saved_model_name = f"{self.model_name}.pth"
        logger.info(f"Save model: {saved_model_name}")
        torch.save({'model_state_dict': self.netG.state_dict(),
                    'trainer_state_dict': self.trainerG.state_dict(),
                    'epoch': epoch}, saved_model_name)

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
        net.eval()
    ret = []
    with torch.no_grad():
        for val_blur, val_sharp in data_iter:
            val_blur = val_blur.to(device)
            val_sharp = val_sharp.to(device)
            ret.append(cal_psnr(net(val_blur), val_sharp))
    return max(ret)

def setup_seed(seed=128):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    logger.info(f"set random seed to {seed}")

if __name__ == '__main__':
    t = Trainer()
    np.seterr(invalid='ignore')
    cnt = 50
    while cnt:
        cnt -= 1
        try:
            t.train()
        except:
            print(traceback.format_exc())
            continue
        else:
            break

    # os.system("shutdown -s")
