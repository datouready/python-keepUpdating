# torch的训练流程
# 1、定义数据集Dataset
# 也可以使用官方提供的torchvision.datasets包的各种标准测试数据、自动下载解压预处理
# 例如mnist、cifar等等，一行代码下载并使用数据集
# 这里我们讲自己实现从文件读取图片的过程

import torchvision.transforms.functional as T

#1、自己定义数据集，需要自己实现len和getitem
class MyDataset:
    def __init__(self,directory):
        self.directory=directory
        self.files=load_files(directory)
    def __len__(self):
        return len(self.files)
    def __getitem__(self,index):
        #处理files[index]，得到image和label
        #在这里我们会对图像做增广等操作
        return T.to_tensor(image),label

#使用方式
dataset=MyDataset()
dataset_length=len(dataset)#这里会自动调用__len__
dataset_item=dataset[10]#这里会自动调用__getitem__

# 2、定义模型结构
    import torch.nn as nn

    class ResNet18(nn.Module):#继承nn.Module
        def __init__(self,num_classes):
            super(ResNet18,self).__init__()#这句话是必带的，初始化父类的初始化函数

            #定义每个层的信息。只要是继承自nn.Module,都会被Module记录下来，然后就会将这些参数管理起来,可以在_modules属性里面查看
            self.conv1=nn.Conv2d(3,self.inplanes,kernek_size=7,stride=2,padding=3)
            self.bn1=norm_layer(self.inplanes)
            #...省略
            self.fc=nn.Linear(512*block.expansion,num_classes)
        
        def forward(self,x):
            #定义模型的前向过程
            x=self.conv1(x)
            x=self.bn1(x)
            # ...省略
            x=x.reshape(x.size(0),-1)
            x=self.fc(x)
            return x
        def __call__(self,x):
            return self.forward(x)
    
    #model=Resnet18(100)
    #output=model(image)#括号这种调用实际上是类似__call__,更直观的是，等价与x=self.conv1(x)之类的操作

    # 3、定义Dataset实例和DataLoader实例
    train_dataset=MyDataset("./train")
    #dataloader实现了多进程数据并行加载，自己实现会很操心，所以使用这个,num_workers进程数，win下是给num_workers=0
    train_dataloader=DataLoader(dataset=train_dataset,batch_size=64,shuffle=True,num_workers=10)

    # 4、定义模型实例
    model=ResNet18(1000)
    model.cuda() #转到cuda,意思放到显卡里面，准备训练,不是创建一个cuda的model，不是model=model.cuda()
    # 显卡控制，一定要在import torch之前
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="3"

    # 5、定义loss函数
    #这里采用交叉熵loss,训练分类器
    loss_function=nn.CrossEntropyLoss()
    #自己写一个交叉熵损失

    # 6、定义优化器optimizer