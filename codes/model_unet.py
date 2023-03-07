# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision.models import resnet50, ResNet50_Weights
from model_swin import swin_l

# #########--------- Components ---------#########
def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    
    print('initialize network with %s' % init_type)
    net.apply(init_func)


class res_conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(res_conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            # shufflenet_v2_x1_0(ch_out, ch_out),
            # nn.ReLU(inplace=True),

        )
        self.downsample = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(ch_out),
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv(x)
        
        return self.relu(out + residual)


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            # nn.Upsample(scale_factor=2),
            # nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(ch_out),
            # nn.ReLU(inplace=True)
            nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        x = self.up(x)
        
        return x


# #########--------- Networks ---------#########
class SRF_UNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(SRF_UNet, self).__init__()

        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        swin = swin_l(img_ch=img_ch)
        self.conv_res = nn.Conv2d(img_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.encoder_res1 = resnet.layer1
        self.encoder_res2 = resnet.layer2
        self.encoder_res3 = resnet.layer3
        self.encoder_res4 = resnet.layer4

        self.conv_swin = swin.features[0:2]

        self.encoder_swin1 = swin.features[2:4]
        self.encoder_swin2 = swin.features[4:6]
        self.encoder_swin3 = swin.features[6:8]

        self.decoder_vessel = nn.Sequential()
        self.decoder_faz = nn.Sequential()
        self.decoder_heatmap = nn.Sequential()

        for ch_in_1, ch_in_2, ch_out_1, ch_out_2 in zip([2048, 1024, 512, 256, 64], [2048, 1024, 512, 128, 64],
                                                        [1024, 512, 256, 64, 64], [1024, 512, 256, 64, 32]):
            self.decoder_vessel.append(up_conv(ch_in=ch_in_1, ch_out=ch_out_1))
            self.decoder_vessel.append(res_conv_block(ch_in=ch_in_2, ch_out=ch_out_2))

            self.decoder_faz.append(up_conv(ch_in=ch_in_1, ch_out=ch_out_1))
            self.decoder_faz.append(res_conv_block(ch_in=ch_in_2, ch_out=ch_out_2))

            self.decoder_heatmap.append(up_conv(ch_in=ch_in_1, ch_out=ch_out_1))
            self.decoder_heatmap.append(res_conv_block(ch_in=ch_in_2, ch_out=ch_out_2))

        self.decoder_vessel.append(nn.Conv2d(32, output_ch, kernel_size=1))
        self.decoder_faz.append(nn.Conv2d(32, output_ch, kernel_size=1))
        self.decoder_heatmap.append(nn.Conv2d(32, output_ch, kernel_size=1))

    def decoder(self, D, x0, x1, x2, x3, x4, down_pad, right_pad):
        # decoding + concat path
        d5 = D[0](x4)
        d5 = torch.cat((x3, d5), dim=1)

        # Decoder
        if down_pad and (not right_pad):
            d5 = d5[:, :, :-1, :]
        if (not down_pad) and right_pad:
            d5 = d5[:, :, :, :-1]
        if down_pad and right_pad:
            d5 = d5[:, :, :-1, :-1]

        d5 = D[1](d5)

        d4 = D[2](d5)
        d4 = torch.cat((x2, d4), dim=1)
        d4 = D[3](d4)

        d3 = D[4](d4)
        d3 = torch.cat((x1, d3), dim=1)
        d3 = D[5](d3)

        d2 = D[6](d3)
        d2 = torch.cat((x0, d2), dim=1)
        d2 = D[7](d2)

        d1 = D[8](d2)
        d1 = D[9](d1)

        d1 = D[-1](d1)
        out = nn.Sigmoid()(d1)

        return out
    
    def forward(self, x):
        # encoding path
        x0 = self.conv_res(x)
        x0 = self.bn(x0)
        x0 = self.relu(x0)
        x1 = self.maxpool(x0)
        
        x1 = self.encoder_res1(x1)
        x2 = self.encoder_res2(x1)
        x3 = self.encoder_res3(x2)

        y1 = self.conv_swin(x)
        y2 = self.encoder_swin1(y1)
        y3 = self.encoder_swin2(y2)
        
        down_pad = False
        right_pad = False
        if x3.size()[2] % 2 == 1:
            x3 = F.pad(x3, (0, 0, 0, 1))
            y3 = y3.permute(0, 3, 1, 2)
            y3 = F.pad(y3, (0, 0, 0, 1))
            y3 = y3.permute(0, 2, 3, 1)
            down_pad = True
        if x3.size()[3] % 2 == 1:
            x3 = F.pad(x3, (0, 1, 0, 0))
            y3 = y3.permute(0, 3, 1, 2)
            y3 = F.pad(y3, (0, 1, 0, 0))
            y3 = y3.permute(0, 2, 3, 1)
            right_pad = True
        
        x4 = self.encoder_res4(x3)
        y4 = self.encoder_swin3(y3)

        x1 = torch.add(x1, y1.permute(0, 3, 1, 2))
        x2 = torch.add(x2, y2.permute(0, 3, 1, 2))
        x3 = torch.add(x3, y3.permute(0, 3, 1, 2))
        x4 = torch.add(x4, y4.permute(0, 3, 1, 2))

        x_vessel, x_faz, x_heatmap = self.decoder(self.decoder_vessel, x0, x1, x2, x3, x4, down_pad, right_pad),\
                                     self.decoder(self.decoder_faz, x0, x1, x2, x3, x4, down_pad, right_pad), \
                                     self.decoder(self.decoder_heatmap, x0, x1, x2, x3, x4, down_pad, right_pad)

        
        return x_vessel, x_faz, x_heatmap, torch.tensor([1, 0, 0, 0], dtype=torch.float)