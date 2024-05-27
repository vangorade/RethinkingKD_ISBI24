import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
# from .UNet_utils import conv_block, up_conv

import torch.nn as nn
import torch


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class UNet_R50(nn.Module):
    """
    UNet with ResNet-50 Encoder
    """
    def __init__(self, in_ch=1, out_ch=4):
        super(UNet_R50, self).__init__()
        
        # Encoder (ResNet-50)
        resnet = models.resnet50(pretrained=False)
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4
        
        # Decoder
        self.Up5 = up_conv(2048, 1024)  # ResNet-50 layer4 output has 2048 channels
        self.Up_conv5 = conv_block(1024 + 1024, 1024)  # Updated input channels
        
        self.Up4 = up_conv(1024, 512)
        self.Up_conv4 = conv_block(512 + 512, 512)  # Updated input channels
        
        self.Up3 = up_conv(512, 256)
        self.Up_conv3 = conv_block(256 + 256, 256)  # Updated input channels
        
        self.Up2 = up_conv(256, 128)
        self.Up_conv2 = conv_block(128 + 64, 128)  # Updated input channels
        
        self.Up1 = up_conv(128, 64)
        self.Up_conv1 = conv_block(64 + 1, 64)  # Updated input channels

        self.Conv = nn.Conv2d(64, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        e1 = F.interpolate(e1, size=d2.size()[2:], mode='bilinear', align_corners=True)

        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Up1(d2)
        d1 = torch.cat((x, d1), dim=1)  # Fix the input channel dimension here
        d1 = self.Up_conv1(d1)

        out = self.Conv(d1)

        return e1, e2, e3, e4, e5, d5, d4, d3, d2, out


# if __name__ == '__main__':

#     model = UNet_R50(in_ch=1, out_ch=4)
#     x = torch.rand(2,1,224,224)
#     e1,e2,e3,e4,e5, d5,d4,d3,d2,out   = model(x)
#     print(e1.shape,e2.shape,e3.shape,e4.shape,e5.shape, d5.shape,d4.shape,d3.shape,d2.shape,out.shape)