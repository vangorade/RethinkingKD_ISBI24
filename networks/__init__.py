from .deeplabv3_plus import DeepLabV3Plus
from .ENet import ENet
from .ERFNet import ERFNet
from .ESPNet import ESPNet
from .mobilenetv2 import MobileNetV2
from .NestedUNet import NestedUNet, NestedUNet_x
from .RAUNet import RAUNet
from .resnet18 import Resnet18, Resnet18_x
from .UNet import U_Net
from .PspNet.pspnet import PSPNet
###################################
from .UNet_R50 import UNet_R50
from .UNet_R18  import UNet_R18
from .UNet_MBNV2 import UNet_MBNV2
from .MALUNet import MALUNet

def get_model(model_name: str, channels: int):
    assert model_name.lower() in ['unet_r50', 'unet_r18', 'unet_mbnv2', "unet++", "resnet18", "mobilenetv2", 'resnet18_x']
    if model_name.lower() == 'unet_r50':
        model = U_Net(in_ch=1, out_ch=channels)
    elif model_name.lower() == 'unet_r18':
        model = UNet_R18(in_ch=1, out_ch=channels)
    elif model_name.lower() == 'unet_mbnv2':
        model = UNet_MBNV2(in_ch=1, out_ch=channels)
    elif model_name.lower() == 'unet++':
        model = NestedUNet(in_ch=1, out_ch=channels)
    elif model_name.lower() == 'resnet18':
        model = Resnet18(num_classes=channels)
    elif model_name.lower() == 'resnet18_x':
        model = Resnet18_x(num_classes=channels)
    elif model_name.lower() == 'mobilenetv2':
        model = MobileNetV2(num_classes=channels)
    return model

