import time

import torch.cuda
import torchvision
# 1.准备数据集
from torch.nn.modules import loss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model_cifar10 import *


# from read_dataset import *
train_dataset = torchvision.datasets.CIFAR10("./data_cifar10", train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())
test_dataset = torchvision.datasets.CIFAR10("./data_cifar10", train=False, download=True,
                                         transform=torchvision.transforms.ToTensor())


# 2.训练数据集长度
train_data_size = len(train_dataset)
test_data_size = len(test_dataset)
# print(train_data_size)
# print(test_data_size)


# 3.DataLoader加载
train_dataloader = DataLoader(train_dataset, batch_size=1)
test_dataloader = DataLoader(test_dataset, batch_size=1)


# 4.搭建神经网络模型
#from model_cifar10 import *
vgg16 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
vgg16.classifier[6] = nn.Linear(4896,2)

# 5.创建神经网络模型:损失函数，优化器
# clara = Clara()
clara = vgg16
if torch.cuda.is_available():
    clara = clara.cuda()

#损失函数
loss.fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss = loss.cuda()

#优化器
learning_rate = 1e-2 #0.01
optimizer = torch.optim.SGD(clara.parameters(), lr=learning_rate)


#6.设置训练网络的参数：记录
total_train_step = 0
total_test_step = 0
epoch = 10

#添加tensorBoard
writer = SummaryWriter("./logs")

#7.开始训练
loss_fn = nn.CrossEntropyLoss()
for i in range(epoch):
    print("-----第{}轮训练开始------".format(i+1))
    start_time = time.time()

    #训练步骤
    for data in train_dataloader:
        optimizer.zero_grad() #梯度归零

        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = clara(imgs)

        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 ==0:
            print("训练次数:{}, Loss:{}".format(total_train_step, loss))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    end_time = time.time()
    print("第{}轮训练时间".format(i+1,end_time-start_time))




    #测试步骤
    total_test_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda
                targets = targets.cuda()

            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss
            accuracy = (outputs.argmax(1)== targets).sum()
            print(accuracy)
            total_accuracy = total_accuracy + accuracy


        total_test_step = total_test_step + 1
        print("整体测试集上的Loss:{}".format(total_test_loss))
        writer.add_scalar("test_loss", total_test_loss, total_test_step)
        writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)

    writer.close()