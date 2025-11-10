import torch
from torch import nn
from torch.nn import L1Loss

inputs = torch.tensor([1,2,3], dtype=torch.float32)
targets = torch.tensor([1,2,5], dtype = torch.float32)

inputs = torch.reshape(inputs, (1,1,1,3))  #bath_size，channel，第一个数字？，数字总和
targets = torch.reshape(targets, (1,1,1,3))

# #mean absolute error
# loss = L1Loss(reduction='mean')  #mean:绝对值相加÷总数 sum：绝对值相加
# result = loss(inputs, targets)
# print(result)

#MSW : mean squared error 方差
loss_MSE = nn.MSELoss()
result = loss_MSE(inputs,targets)
print(result)

#CrossEntropyLoss 交叉删
