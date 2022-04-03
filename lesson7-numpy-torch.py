#numpy是管cpu操作，在cpu上做矩阵运算，读取数据，就是读取到内存里面，对文件图像做各种操作，这时候numpy作用就很大
#torch是管gpu操作,torch具有自动微分功能
import numpy as np

a=np.array([1,2,3],dtype=np.float32)#对类型要心里有底,numpy和torch默认类型不一样
print(a,type(a),a.dtype,a.shape)

#全为0
b=np.zeros((3,3),dtype=np.float32)

#全为1
c=np.ones((3,3)).astype(np.float32) #转换类型

# 随机数，正态分布
d=np.random.randn(3,3)
w=d[:,:2]
d.argmax(axis=0)#返回一个列中最大的数所在的行数 位置，在torch里面使用dim

#产生序列
np.arange(1,10)#array([1,2,3,4,56,7,8,9,10])

e=d.copy()#复制一份

#数组索引选择
k=e[[0,2],:]#选取第0行，第2行所有列

# torch.tensor基本操作
#具有numpy的扩展功能，很多numpy没有实现的，tensor也实现了，torch主要在GPU，当然也支持在cpu上

import torch

a=torch.tensor([1,2,3])
print(a,a.shape,a.dtype)