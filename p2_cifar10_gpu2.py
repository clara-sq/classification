import torchvision
# 1.准备数据集
from torch.nn.modules import loss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model_cifar10 import *

device = torch.device("cuda")

#1.设置数据
train_data = torchvision.datasets.CIFAR10("./data_cifar10", train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10("./data_cifar10", train=False, download=True,
                                         transform=torchvision.transforms.ToTensor())


# 2.训练数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
# print(train_data_size)
# print(test_data_size)


# 3.DataLoader加载
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


# 4.搭建神经网络模型
#from model_cifar10 import *



# 5.创建神经网络模型:损失函数，优化器
clara = Clara()
clara = clara.to(device)
#损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
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

    #训练步骤
    for data in train_dataloader:
        optimizer.zero_grad() #梯度归零

        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = clara(imgs)

        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 ==0:
            print("训练次数:{}, Loss:{}".format(total_train_step, loss))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # end_time = time.time()
    # print("第{}轮训练时间".format(i+1,end_time-start_time))



    #测试步骤
    total_test_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device
                                 )
            outputs = clara(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss

        total_test_step = total_test_step + 1
        print("整体测试集上的Loss:{}".format(total_test_loss))
        writer.add_scalar("test_loss", total_test_loss, total_test_step)


    writer.close()