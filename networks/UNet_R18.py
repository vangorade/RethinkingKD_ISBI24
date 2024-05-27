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


class UNet_R18(nn.Module):
    def __init__(self, in_ch=1, out_ch=4):
        super(UNet_R18, self).__init__()
        
        # Encoder (ResNet-18)
        resnet = models.resnet18(pretrained=False)
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
        self.Up5 = up_conv(512, 256)  # ResNet-18 layer4 output has 512 channels
        self.Up_conv5 = conv_block(256 + 256, 256)
        
        self.Up4 = up_conv(256, 128)
        self.Up_conv4 = conv_block(128 + 128, 128)
        
        self.Up3 = up_conv(128, 64)
        self.Up_conv3 = conv_block(64 + 64, 64)
        
        self.Up2 = up_conv(64, 32)
        self.Up_conv2 = conv_block(64+32, 32)
        
        self.Conv = nn.Conv2d(32, out_ch, kernel_size=1, stride=1, padding=0)
    
    def t_guided_s(self, s, t):
        """
        Compact Cross-Attention from Teacher to Student feature maps.
        """
#         print(s.shape)
        s_pri = s
        channel_decomp = nn.Conv2d(s.shape[1], t.shape[1], kernel_size=1).cuda()
        s = channel_decomp(s)
        
        if s.shape[2] != t.shape[2]:
            s = F.interpolate(s, t.size()[-2:], mode='bilinear')
                
        attn_map = torch.matmul(t, s.transpose(2, 3))
        attn_map = F.softmax(attn_map, dim=-1)
        guided_s = torch.matmul(attn_map.transpose(2, 3), s)
        
        channel_comp = nn.Conv2d(guided_s.shape[1], s_pri.shape[1], kernel_size=1).cuda()
        guided_s = channel_comp(guided_s)
        
        guided_s = F.interpolate(guided_s, s_pri.size()[-2:], mode='bilinear')
        
        return guided_s

    def s_guided_t(self, t, s):
        """
        Compact Cross-Attention from Student to Teacher feature maps.
        """
        attn_map = torch.matmul(s, t.transpose(2, 3))
        attn_map = F.softmax(attn_map, dim=-1)
        guided_t = torch.matmul(attn_map.transpose(2, 3), t)
        return guided_t
    
    def forward(self, x, t_attn_skips):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        if t_attn_skips is None:
            t_s_e1, t_s_e2, t_s_e3, t_s_e4, t_s_e5 = e1, e2, e3, e4, e5

        else:
            
            t_e1, t_e2, t_e3, t_e4, t_e5 = t_attn_skips
            # print((e1.shape), (t_e1.shape))
            t_s_e1, t_s_e2, t_s_e3, t_s_e4, t_s_e5 = self.t_guided_s(e1, t_e1), self.t_guided_s(e2, t_e2), self.t_guided_s(e3, t_e3), self.t_guided_s(e4, t_e4), self.t_guided_s(e5, t_e5)
       
        d5 = self.Up5(t_s_e5)
        d5 = torch.cat((t_s_e4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((t_s_e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((t_s_e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        t_s_e1 = F.interpolate(t_s_e1, size=d2.size()[2:], mode='bilinear', align_corners=True)
        d2 = torch.cat((t_s_e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)
        out = F.interpolate(out, size=x.size()[2:], mode='bilinear', align_corners=True)

        return t_s_e1, t_s_e2, t_s_e3, t_s_e4, t_s_e5, d5, d4, d3, d2, out


if __name__ == '__main__':
    from UNet import U_Net
    model_t = U_Net(in_ch=1, out_ch=4)
    x = torch.rand(2,1,224,224)
    e1,e2,e3,e4,e5, d5,d4,d3,d2,out   = model_t(x)
    print(e1.shape,e2.shape,e3.shape,e4.shape,e5.shape, d5.shape,d4.shape,d3.shape,d2.shape,out.shape)

    model = UNet_R18(in_ch=1, out_ch=4)
    x = torch.rand(2,1,224,224)
    e1,e2,e3,e4,e5, d5,d4,d3,d2,out   = model(x, [e1,e2,e3,e4,e5])
    print(e1.shape,e2.shape,e3.shape,e4.shape,e5.shape, d5.shape,d4.shape,d3.shape,d2.shape,out.shape)