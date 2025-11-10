import os.path
from tkinter import Image

import torch
import torchvision.transforms
from rsa import transform
from torch.utils.data import Dataset

data_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32,32)),
    torchvision.transforms.ToTensor()
])

class myData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(root_dir, label_dir)
        self.img_list = os.listdir(self.path)
        # print(self.img_list)
        self.transforms = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_name = self.img_list[index]
        img_item_path = os.path.join(self.root_dir, self.label_dir)
        img = Image.open(img_item_path)
        img = self.transforms(img)
        label = torch.tensor(int(self.label_dir))




train_root_dir = r".\datasets\ant_bee\train"
train_ants_label_dir = "0"
train_bees_label_dir = "1"
train_ants_dataset = myData(train_root_dir, train_ants_label_dir, transform=data_transform)
train_bees_dataset = myData(train_root_dir, train_bees_label_dir, transform=data_transform)
train_dataset = train_ants_dataset + train_bees_dataset

test_root_dir = r".\datasets\ant_bee\train"
test_ants_label_dir = "0"
test_bees_label_dir = "1"
test_ants_dataset = myData(test_root_dir, test_ants_label_dir, transform=data_transform)
test_bees_dataset = myData(test_root_dir, test_bees_label_dir, transform=data_transform)
test_dataset = test_ants_dataset + test_bees_dataset


if __name__ == '__main__':
    print("train_dataset:{}".format(len(train_dataset)))
    print("test_dataset:{}".format(len(test_dataset)))
    for data in train_dataset:
        img, target = data
        print(img)
        print(target)