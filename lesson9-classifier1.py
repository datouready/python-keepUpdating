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
        # self.files=load_files(directory)
        self.files=10
    def __len__(self):
        # return len(self.files)
        return 10
    def __getitem__(self,index):
        #处理files[index]，得到image和label
        #在这里我们会对图像做增广等操作
        image=cv2.imread(self.files[index])
        # return T.to_tensor(image),label
        return (10,10)

#使用方式
dataset=MyDataset("./")
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
'''
使用官方的实现
import torch
cross_entropy=torch.nn.CrossEntropyLoss()
predict=torch.tensor([0.1,0.1,0.5]).reshape(1,3).requires_grad_(True)
ground_truth=torch.tensor([2])
print(predict.shape,ground_truth.shape)

loss=cross_entropy(predict,ground_truth)
loss.backward()
predict.grad

#自己实现一个试试
predict=torch.tensor([[0.1,0.1,0.5]]).requires_grad_(True)
ground_thruth_one_hot=torch.tensor([[0,0,1]])
loss2=-torch.sum(torch.log(torch.softmax(predict,1)[ground_truth_one_hot >0]))
loss2.backward()
predict.grad

'''
# 6、定义优化器optimizer
#这里使用SGD,自己可以去了解Adam等其他优化器
op=torch.optim.SGD(model.parameters(),lr=1e-3,momentum=0.9,weight_decay=1e-5)

# 7、循环执行优化过程

lr_schedule={
    10:1e-4,#迭代10次使用这个
    20:1e-5,
}
total_epoch=30
for epoch in range(total_epoch):
    #根据迭代轮数设置学习率
    if epoch in lr_schedule:
        new_lr=lr_schedule[epoch]
        #得到学习率之后，使用下面方式设置学习率，很经典的办法，一定记住 
        for param_group in op.param_groups:
            param_group["lr"]=new_lr
    
    #迭代一轮
    for index_batch,(image,labels) in enumerate(train_dataloader):
        #image.shape, 64,3,300,300 batch_size C H W
        #labels.shape,64,1
        images=images.cuda()
        labels=labels.cuda()
        output=model(images)
        loss=loss_function(output,labels)

        #优化模型三部曲
        #1、梯度清空
        op.zero_grad()
        # 2、loss函数执行反向计算梯度
        loss.backward()

        #3、应用梯度并进行更新
        op.step()

        print("epoch{},loss{}".format(epoch,loss.item))

# 8、模型的测试
#定义模型实例
model=ResNet18(1000)
model.cuda()#转到cuda
model.eval()#进入评估模式，此时model(image)不会进行变量跟踪，
#相应的，如果训练中做评估，则可以通过eval到评估模式，评估完毕得到精度后，model.train()再次进入训练模式

#数据处理，这里的变换操作与dataset的定义一致
tran=torchvision.transforms.Compose([
    torchvision.transforms.Reszie(256),
    torchvision.transforms.ToTensor(),
    torchvision.Normalzie([0.485,0.456,0.406],[0.229,0.224,0.225])
])

#禁止模型执行过程中为计算梯度而使用更多显存存储中间过程
with torch.no_grad():
    for file in files:
        #加载图像
        image=cv2.imread(file)

        #图像做归一化处理
        image=train(image)
        
        #预测得到结果
        outputs=model(image)
        predict_label=output.argmax(dim=1).cpu()

        #这里的 predict_label 即是输出结果

# 9、模型的保存和加载
#如果是多GPU训练，请搜索相关资料，此时保存和加载是有讲究的
#保存
torch.save(model.state_dict(),"my_model.pth")

#加载
checkpoint=torch.load("my_model.pth")
model.load_state_dict(checkpoint)