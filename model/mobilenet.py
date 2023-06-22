import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small


def mobilenetv3():
    return mobilenet_v3_small()

class MobileNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()