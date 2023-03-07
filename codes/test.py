from codes.model_swin import swin_l
from model_swin import *
from torchinfo import summary
from model_unet import SRF_UNet
import torch.nn.functional as F
import torch

# swin_model = swin_l(img_ch=2)
# summary(swin_model, (1, 2, 304, 304))
# print(swin_model.features[0])
model = SRF_UNet(img_ch=2)
summary(model, (1, 2, 304, 304))

y3 = torch.randn(1, 19, 19, 1024)

y3 = y3.permute(0, 3, 1, 2)
y3 = F.pad(y3, (0, 0, 0, 1))
y3 = y3.permute(0, 2, 3, 1)

print("y3.shape:", y3.shape)

# import torch
#
# x = torch.randn(1, 512, 38, 38)
# x = x.permute(0, 2, 3, 1)
# print(x.shape)
# x = x.permute(0, 3, 1, 2)
# print(x.shape)