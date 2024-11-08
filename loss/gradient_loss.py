import torch
import torch.nn.functional as F
import torch.nn as nn

class GradientLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.l1loss = nn.L1Loss()
        self.weight = weight
        self.register_buffer("kernel_x",
                             torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]],dtype=torch.float).repeat(3,1,1,1))
        self.register_buffer("kernel_y",
                             torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]],dtype=torch.float).repeat(3,1,1,1))

    def gradient(self,img):
        grad_x = F.conv2d(img, self.kernel_x, groups=3)
        grad_y = F.conv2d(img, self.kernel_y, groups=3)
        return grad_x, grad_y

    def gradient_loss(self,input, target, loss_func):
        grad_x = self.gradient(input)
        grad_y = self.gradient(target)
        return sum(loss_func(ig, tg) for ig, tg in zip(grad_x, grad_y)) / len(grad_x)

    def forward(self, input, target):
        l1_loss = self.l1loss(input,target)
        grad_loss = self.gradient_loss(input,target,F.l1_loss)
        return l1_loss + grad_loss * self.weight