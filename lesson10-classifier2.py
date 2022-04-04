##torch 分类器训练
# RuntimeError: Input type (torch.FloatTensor)(这里意思数据在CPU上) and weight type (torch.cuda.FloatTensor) （模型在GPU上）
# should be the same or input should be a MKLDNN tensor and weight is a dense tensor

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# inputs = inputs.to(device)
# inputs= input.to('cuda')

# 对于nn.Module:

# model = model.cuda() 
# model.cuda() 
# 上面两句能够达到一样的效果，即对model自身进行的内存迁移。

# 对于Tensor:

# 和nn.Module不同，调用tensor.cuda()
# 只是返回这个tensor对象在GPU内存上的拷贝，而不会对自身进行改变。
# 因此必须对tensor进行重新赋值，即tensor=tensor.cuda().

#设置使用的GPU是0号，请改成自己能用的GPUID号，一定要在import torch之前
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import shutil          #文件操作相关
import torch
import torch.nn as nn
import torchvision.transforms as T 
import math            #数学相关
import cv2              #opencv
import random           #随机数
import torchvision.models #标准模型
from torch.utils.data import DataLoader            #数据加载
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

print(torch.cuda.is_available())

# class LDataset():
#     def __init__(self,directory):
#         self.directory=directory
#         self.files=os.listdir(directory)
#     def __len__(self):
#         return len(self.files)
#     def __getitem__(self,index):
#         file=self.files[index]
#         image=cv2.imread(f"{self.directory}/{file}")
#         image=cv2.resize(image,(32,32))
#         # print(image.shape)
#         label=int(file[0])
#         return T.functional.to_tensor(image),label  # H,W,C   C,H,W  torch.tensor

class LDataset:

    def __init__(self, directory):

        self.directory = directory
        files = os.listdir(directory)

        # random shuffle files，keep 1000 files
        random.shuffle(files)

        files = files[:1000]
        labels = [int(name[0]) for name in files]

        self.files = list(zip(files, labels))

        # 定义数据变换方法序列，进行图像增广
        self.trans = T.Compose([
            T.ToPILImage(),#先变成PIL
            T.Resize(32),
            T.RandomGrayscale(),#随机灰度缩放
            T.RandomRotation(25),#随机在25度旋转
            T.ToTensor(),
            T.Normalize([0.5], [1])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        file, label = self.files[index]
        image = cv2.imread(f"{self.directory}/{file}")

        # augment and normalization image
        image = self.trans(image)
        #image = (image / 255 - 0.5).astype(np.float32)

        #这里省略增广方法
        #return T.functional.to_tensor(image), label
        return image, label

#dataset=LDataset("train")
#len(dataset),dataset[0][0].shape

# fs=os.listdir("train")
# image=cv2.imread("image.png")
# plt.imshow(image)

#定义我们自己的网络结构
# class LeNet5(nn.Module):
#     def __init__(self,num_classes):
#         super(LeNet5,self).__init__()

#         #nn.Sequential可以简化以下代码，对于顺序下去的话
#         self.conv1=nn.Conv2d(3,6,kernel_size=5,stride=1,)
#         self.pool1=nn.MaxPool2d(2,stride=2)
#         self.conv2=nn.Conv2d(6,16,5)
#         self.pool2=nn.MaxPool2d(2,stride=2)
#         self.conv3=nn.Conv2d(16,120,5)
#         self.fc1=nn.Linear(120,84)
#         self.fc2=nn.Linear(84,num_classes)
#     def forward(self,x):
#         print(x.shape)
#         x=self.conv1(x)
#         print(x.shape)
#         x=self.pool1(x)
#         x=self.conv2(x)
#         x=self.pool2(x)
#         x=self.conv3(x)
#         x=x.view(x.size(0),-1)          #将batch维度留下，其它的变为一个维度
#         x=self.fc1(x)
#         x=self.fc2(x)
#         return x

class LeNet5(nn.Module):

    def __init__(self, num_classes):
        super(LeNet5, self).__init__()

        # nn.Sequential可以简化以下代码，对于顺序下去的话
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

#定义resnet18的网络结构，一定要根据模型来改，了解模型
class ResNet18(nn.Module):
    def __init__(self,num_classes):
        super(ResNet18,self).__init__()
        self.extract=torchvision.models.resnet18(pretrained=False)
        fc_features=self.extract.fc.in_features
        self.extract.fc=nn.Linear(fc_features,num_classes)
    def forward(self,x):
        return self.extract(x)

#定义数据集实例
train_dataset=LDataset("../../datasets/train")
train_dataloader=DataLoader(dataset=train_dataset,batch_size=10,shuffle=True,num_workers=4)

#获取模型实例
model=ResNet18(10)

model.to(device)

#定义损失函数和优化器
loss_function=nn.CrossEntropyLoss()
op=torch.optim.SGD(model.parameters(),lr=1e-2,momentum=0.9)

lr_schedule={
    15:1e-3,
    20:1e-4,
}

total_epoch=1

for epoch in range(total_epoch):
    
    #根据迭代轮数设置学习率
    # if epoch in lr_schedule:
    #     new_lr=lr_shcedule[epoch]
    #     for param_group in op.param_groups:
    #         param_group["lr"]=new_lr

    #迭代一轮
    #item=next(iter(dataloader))
    for index_batch,(images,labels) in enumerate(train_dataloader):
        #转换成cuda()
        images=images.to(device)
        labels=labels.to(device)
        # print(images.dtype)
        output=model(images)
        # print(output.dtype)
        loss=loss_function(output,labels)

        op.zero_grad()
        loss.backward()
        op.step()
        # print(f"opech:{epoch},loss:{loss.item()}")

    print(f"opech:{epoch},loss:{loss.item()}")
