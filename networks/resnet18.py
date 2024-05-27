import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class Resnet18(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(Resnet18, self).__init__()
        model = resnet18(pretrained)

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.last_conv = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[-2:]
        x = torch.cat([x, x, x], dim=1)  # 扩充为3通道
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        s_e1 = x
        s_d2 = F.interpolate(s_e1, size=input_size, mode='bilinear', align_corners=True)
        x = self.maxpool(x)

        x = self.layer1(x)
        s_e2 = x
        s_d3= F.interpolate(s_e2, size=input_size, mode='bilinear', align_corners=True)
        x = self.layer2(x)
        s_e3 = x
        s_d4 = F.interpolate(s_e3, size=input_size, mode='bilinear', align_corners=True)
        x = self.layer3(x)
        s_e4 = x
        s_d5 = F.interpolate(s_e4, size=input_size, mode='bilinear', align_corners=True)
        x = self.layer4(x)
        s_e5 = x
        x = self.last_conv(x)
        s_output = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        return s_e1, s_e2, s_e3, s_e4, s_e5, s_d5, s_d4, s_d3, s_d2, s_output



class Resnet18_x(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(Resnet18_x, self).__init__()
        model = resnet18(pretrained)

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.last_conv = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[-2:]
        x = torch.cat([x, x, x], dim=1)  # 扩充为3通道
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        s_e1 = x
        s_d2 = F.interpolate(s_e1, size=input_size, mode='bilinear', align_corners=True)
        x = self.maxpool(x)

        x = self.layer1(x)
        s_e2 = x
        s_d3= F.interpolate(s_e2, size=input_size, mode='bilinear', align_corners=True)
        x = self.layer2(x)
        s_e3 = x
        s_d4 = F.interpolate(s_e3, size=input_size, mode='bilinear', align_corners=True)
        x = self.layer3(x)
        s_e4 = x
        s_d5 = F.interpolate(s_e4, size=input_size, mode='bilinear', align_corners=True)
        x = self.layer4(x)
        s_e5 = x
        x = self.last_conv(x)
        s_output = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        return s_d3
