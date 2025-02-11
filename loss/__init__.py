from .gradient_loss import GradientLoss
from .clamp_loss import ClampLoss
from .lbp_loss import YL1LBP,YLBP
from .lpips_loss import LPIPSWith
from .discriminator_loss import DiscriminatorBCELoss,DiscriminatorHingeLoss

__all__ = ["GradientLoss",
           "ClampLoss",
           "YL1LBP","YLBP",
           "LPIPSWith",
           "DiscriminatorBCELoss","DiscriminatorHingeLoss"]