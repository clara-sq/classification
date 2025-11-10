import torchvision
from torch import nn

vgg16_true = torchvision.models.vgg16(weights = torchvision.models.VGG16_Weights.DEFAULT)
vgg_false = torchvision.models.vgg16()

vgg16_true.classifier[6] = nn.Linear(4096,2)

print(vgg_false)