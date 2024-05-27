import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

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

class UNet_MBNV2(nn.Module):
    """
    UNet with MobileNetV2 Encoder
    """
    def __init__(self, in_ch=3, out_ch=4):
        super(UNet_MBNV2, self).__init__()

        # Encoder (MobileNetV2)
        mobilenet_v2 = models.mobilenet_v2(pretrained=True)
        # print(mobilenet_v2)
        mobilenet_v2.features[0][0] = nn.Conv2d(in_ch, 32, kernel_size=3, stride=2, padding=1, bias=False)
        print(mobilenet_v2)
        self.encoder1 = mobilenet_v2.features[0:2]
        self.encoder2 = mobilenet_v2.features[2:4]
        self.encoder3 = mobilenet_v2.features[4:7]
        self.encoder4 = mobilenet_v2.features[7:14]
        self.encoder5 = mobilenet_v2.features[14:18]


        # Decoder
        self.Up5 = up_conv(320, 96)  # MobileNetV2 last layer output has 320 channels
        self.Up_conv5 = conv_block(96 + 96, 96)  # Adjust input channels to match MobileNetV2

        self.Up4 = up_conv(96, 24)
        self.Up_conv4 = conv_block(24 + 32, 24)  # Adjust input channels to match MobileNetV2

        self.Up3 = up_conv(24, 16)
        self.Up_conv3 = conv_block(16 + 24, 16)  # Adjust input channels to match MobileNetV2

        self.Up2 = up_conv(16, 8)
        self.Up_conv2 = conv_block(8 + 16, 8)  # Adjust input channels to match MobileNetV2

        self.Up1 = up_conv(8, 3)
        self.Up_conv1 = conv_block(3 + 1, 1)  # Adjust input channels to match MobileNetV2

        self.Conv = nn.Conv2d(1, out_ch, kernel_size=1, stride=1, padding=0)
    
    def t_guided_s(self, s, t):
        """
        Compact Cross-Attention from Teacher to Student feature maps.
        """
#         print(s.shape)
        s_pri = s
        channel_decomp = nn.Conv2d(s.shape[1], t.shape[1], kernel_size=1)
        s = channel_decomp(s)
        
        if s.shape[2] != t.shape[2]:
            s = F.interpolate(s, t.size()[-2:], mode='bilinear')
                
        attn_map = torch.matmul(t, s.transpose(2, 3))
        attn_map = F.softmax(attn_map, dim=-1)
        guided_s = torch.matmul(attn_map.transpose(2, 3), s)
        
        channel_comp = nn.Conv2d(guided_s.shape[1], s_pri.shape[1], kernel_size=1)
        guided_s = channel_comp(guided_s)
        
        guided_s = F.interpolate(guided_s, s_pri.size()[-2:], mode='bilinear')
        
        return guided_s
    
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
            t_s_e1, t_s_e2, t_s_e3, t_s_e4, t_s_e5 = self.t_guided_s(e1, t_e1), self.t_guided_s(e2, t_e2), self.t_guided_s(e3, t_e3), self.t_guided_s(e4, t_e4), self.t_guided_s(e5, t_e5)
       
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

        return t_s_e1, t_s_e2, t_s_e3, t_s_e4, t_s_e5, d5, d4, d3, d2, out

# r18 = U_Net_MobileNetV2()
# x = torch.rand(2,3,224,224)
# e1,e2,e3,e4,e5, d5,d4,d3,d2,out = r18(x, [e1,e2,e3,e4,e5])
# print(e1.shape,e2.shape,e3.shape,e4.shape,e5.shape, d5.shape,d4.shape,d3.shape,d2.shape,out.shape)
if __name__ == '__main__':
    from UNet import U_Net
    model_t = U_Net(in_ch=1, out_ch=4)
    x = torch.rand(2,1,224,224)
    e1,e2,e3,e4,e5, d5,d4,d3,d2,out = model_t(x)
    print(e1.shape,e2.shape,e3.shape,e4.shape,e5.shape, d5.shape,d4.shape,d3.shape,d2.shape,out.shape)

    model = UNet_MBNV2(in_ch=1, out_ch=4)
    x = torch.rand(2,1,224,224)
    e1,e2,e3,e4,e5, d5,d4,d3,d2,out   = model(x, [e1,e2,e3,e4,e5])
    print(e1.shape,e2.shape,e3.shape,e4.shape,e5.shape, d5.shape,d4.shape,d3.shape,d2.shape,out.shape)