import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from .sophia import SophiaG
from src.core import register


__all__ = ['AdamW', 'SGD', 'Adam', 'MultiStepLR', 'CosineAnnealingLR', 'OneCycleLR', 'LambdaLR']



SGD = register(optim.SGD)
Adam = register(optim.Adam)
AdamW = register(optim.AdamW)

# sophiag = SophiaG(model.parameters(), lr=2e-4, betas=(0.965, 0.99), rho=0.01, weight_decay=0.0025)


MultiStepLR = register(lr_scheduler.MultiStepLR)
CosineAnnealingLR = register(lr_scheduler.CosineAnnealingLR)
OneCycleLR = register(lr_scheduler.OneCycleLR)
LambdaLR = register(lr_scheduler.LambdaLR)