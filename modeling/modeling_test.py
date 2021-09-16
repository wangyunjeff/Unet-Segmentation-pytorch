import os
from torchsummary import summary
import torch

from modeling.unet import Unet
if __name__ == '__main__':
    a = os.getcwd()
    print(a)
    # inputs = torch.randn(2,3,512,512).cuda()
    # model = Unet(num_classes=5).train().cuda()
    # outputs = model(inputs)
