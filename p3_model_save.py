import torch
import torchvision
from torchvision.models import vgg16, VGG16_Weights

#方法1
# vgg16_true = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)
# print(model)

#方法2
# torch.save(vgg16_true, "vgg16_method1.pth")
# torch.save(vgg16_true.state_dict(), "vgg16_method2.pth")