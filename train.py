import math
import numpy as np
import torch
from utils import logger, set_log_name
import utils
from torch.optim.adamw import AdamW
import os
import torch.nn as nn
from nets import SPAN, UNetResA, SRVGGNetCompact, UpUNetResA, UNetDiscriminator
from data import dataset
from loss import DiscriminatorHingeLoss
import random
import traceback


class Trainer:
    def __init__(self):

        self.scale = 1
        self.batch_size = 32
        self.crop_size = 128
        self.dataset = r"E:\Encode\Dataset\YAPD"
        self.device = torch.device("cuda:0")
        self.use_GAN = True

        self.netG = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=8, num_conv=8, upscale=1).to(self.device)
        # self.netG = UpUNetResA(3,3,2,[32,64,128,256],[4,4,4,4],True).to(self.device)
        self.netD = UNetDiscriminator(3, 64).to(self.device) if self.use_GAN else None

        self.model_name = self.netG.get_model_name()

        self.epochs = 1000
        self.cur_epoch = 0
        self.restart_epoch = False
        self.lrG = 1e-4
        self.lrD = 1e-4
        # self.trainerG = Adam(self.netG.parameters(), lr=self.lr)
        self.trainerG = AdamW(self.netG.parameters(), lr=self.lrG, weight_decay=0.001)
        self.trainerD = AdamW(self.netD.parameters(), lr=self.lrD, weight_decay=0.001) if self.use_GAN else None

        self.load_model()
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.trainerG, milestones=[200, 400, 600, 800], gamma=0.5,
                                                              last_epoch=self.cur_epoch - 1)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.trainerG,T_0=10,eta_min=self.lr * 5e-3)
        # self.update_lr()
        self.train_iter, self.test_iter = dataset.load_data(self.batch_size, self.crop_size, self.crop_size, self.scale,
                                                            self.dataset)
        self.total_step = len(self.train_iter)

        set_log_name(self.model_name)

    def train(self):
        setup_seed()
        content_criterion = nn.L1Loss().to(self.device)
        discriminator_criterion = DiscriminatorHingeLoss().to(self.device) if self.use_GAN else None
        self.netG.train()
        logger.info(f"NETWORK:{self.model_name}, SCALE:{self.scale}")
        logger.info("Start training!")

        for epoch in range(self.cur_epoch + 1, self.epochs + 1):
            logger.info(f"EPOCH:{epoch} LEARNING_RATE:{self.trainerG.state_dict()['param_groups'][0]['lr']}")
            PSNR_list = []
            for iteration, data in enumerate(self.train_iter):
                img_lq, img_gt = data
                img_lq = img_lq.to(self.device)
                img_gt = img_gt.to(self.device)

                if not self.use_GAN:
                    self.trainerG.zero_grad()
                    img_lq_output = self.netG(img_lq)
                    content_loss = content_criterion(img_lq_output, img_gt)
                    temp_PSNR = cal_psnr(img_lq_output, img_gt)
                    if epoch > 1 and temp_PSNR < 0:
                        logger.info("psnr error! Try to restart training.")
                        raise ValueError("psnr error! try to restart training.")
                    PSNR_list.append(temp_PSNR)

                    generator_loss = content_loss
                    generator_loss.backward()
                    self.trainerG.step()

                else:
                    # generator step
                    self.netD.requires_grad_(False)
                    self.trainerG.zero_grad()
                    img_lq_output = self.netG(img_lq)
                    content_loss = content_criterion(img_lq_output, img_gt)
                    temp_PSNR = cal_psnr(img_lq_output, img_gt)
                    if epoch > 1 and temp_PSNR < 0:
                        logger.info("psnr error! Try to restart training.")
                        raise ValueError("psnr error! try to restart training.")
                    PSNR_list.append(temp_PSNR)


                    fake_pred_img_lq = self.netD(img_lq_output)
                    discriminator_loss = discriminator_criterion(fake_pred_img_lq)
                    generator_loss = content_loss * 10 + discriminator_loss
                    generator_loss.backward()
                    self.trainerG.step()

                    # discriminator step
                    self.netD.requires_grad_(True)
                    self.trainerD.zero_grad()
                    fake_pred_img_lq = self.netD(img_lq_output.detach())
                    real_pred_img_gt = self.netD(img_gt)
                    discriminator_loss = discriminator_criterion(real_pred_img_gt,fake_pred_img_lq)
                    discriminator_loss.backward()
                    self.trainerD.step()

                if iteration % 80 == 0:
                    logger.info(
                        f"[TRAIN_PSNR {temp_PSNR:.4f}] [LR_G {self.trainerG.state_dict()['param_groups'][0]['lr']}] [LOSS_G] {content_loss:.4f} [LR_D {self.trainerD.state_dict()['param_groups'][0]['lr']}] [LOSS_D] {content_loss:.4f}")

            self.scheduler.step()
            self.cur_epoch += 1
            train_psnr = np.mean(PSNR_list)
            val_psnr = evaluate_accuracy(self.netG, self.test_iter, self.device)

            self.save_model(epoch, False)
            if epoch % 20 == 0:
                self.save_model(epoch, True)
            logger.info(
                f"[EPOCH {epoch}] [TRAIN_PSNR {train_psnr:.4f}] [VAL_PSNR {val_psnr:.4f}] [LR_G {self.trainerG.state_dict()['param_groups'][0]['lr']}] [LR_D {self.trainerD.state_dict()['param_groups'][0]['lr']}]")

    def load_model(self):
        if os.path.exists(self.model_name + ".pth"):
            logger.info(f"Load G model: {self.model_name}")
            checkpointG = torch.load(self.model_name + ".pth")
            self.netG.load_state_dict(checkpointG['model_state_dict'])
            self.trainerG.load_state_dict(checkpointG['trainer_state_dict'])
            self.cur_epoch = checkpointG['epoch']
            logger.info(
                f"Start from epoch: {self.cur_epoch}, learning_rate: {self.trainerG.state_dict()['param_groups'][0]['lr']}")

        if self.use_GAN and os.path.exists(f"{self.netD.get_model_name()}.pth"):
            logger.info(f"Load D model: {self.model_name}")
            checkpointD = torch.load(self.model_name + ".pth")
            self.netD.load_state_dict(checkpointD['model_state_dict'])
            self.trainerD.load_state_dict(checkpointD['trainer_state_dict'])
            logger.info(f"learning_rateD: {self.trainerD.state_dict()['param_groups'][0]['lr']}")

        if self.restart_epoch:
            self.cur_epoch = 0
            logger.info("Restart epoch!")

    def save_model(self, epoch, check=False):
        if check:
            saved_model_name = f"{self.model_name}_epoch{epoch}.pth"
        else:
            saved_model_name = f"{self.model_name}.pth"
        logger.info(f"Save model: {saved_model_name}")
        torch.save({'model_state_dict': self.netG.state_dict(),
                    'trainer_state_dict': self.trainerG.state_dict(),
                    'epoch': epoch}, saved_model_name)
        if self.use_GAN:
            torch.save({'model_state_dict': self.netD.state_dict(),
                        'trainer_state_dict': self.trainerD.state_dict()}, f"{self.netD.get_model_name()}.pth")
            logger.info(f"Save model: {self.netD.get_model_name()}.pth")

    def update_lr(self):
        for p in self.trainerG.param_groups:
            p['lr'] = self.lrG

        if self.use_GAN:
            for p in self.trainerD.param_groups:
                p['lr'] = self.lrD


def cal_psnr(img1, img2):
    n = img1.numel()
    mse = math.fabs(torch.sum((img1 - img2) ** 2) / n)
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
