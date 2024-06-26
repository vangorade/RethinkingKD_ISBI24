"""
Count the MACs/FLOPs/parameters of PyTorch models.
Installation: pip install flopco-pytorch
"""

# import sys
# sys.path.append("../networks/") #here /../ denotes the full path


from flopco import FlopCo
from torchvision.models import resnet50
from networks.ENet import ENet
from networks.ERFNet import ERFNet
from networks.ESPNet import ESPNet
from networks.UNet import U_Net
from networks.RAUNet import RAUNet
from networks.NestedUNet import NestedUNet
from networks.mobilenetv2 import MobileNetV2
from networks.resnet18 import Resnet18
from networks.deeplabv3_plus import DeepLabV3Plus
from networks.PspNet.pspnet import PSPNet
from networks.UNet_R50 import UNet_R50
from networks.UNet_R18  import UNet_R18
from networks.UNet_MBNV2  import UNet_MBNV2

channels = 2
device = 'cuda'
model = UNet_MBNV2(num_classes=2).to(device)

# Estimate model statistics by making one forward pass througth the model,
# for the input image of size 3 x 224 x 224

stats = FlopCo(model, img_size = (2, 1, 384, 384), device = device)

print(stats.total_macs, stats.total_flops)