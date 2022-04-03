import os
import sys#import使用
print(os.listdir("."))#查看当前路径下文件
#基本数据类型
#int,float,str,list,dict,set,tuple
a=123
a=int("123")
b=123.
b=float("123")
print(a,b)

c="abc"
c='abc'
d=list()
d=[]
e=dict()
e={}#hashmap
f=set()
g=tuple()

#print输出
a="abc"
a="age is %d"%15 #固定格式
a="age is {}".format(15)
age=150
a=f"age is {age}"#f-string效率最高，一般用这种
a=r"c:\a\b\c\d"#不会转义 raw-string

#list
b=[1,2,3,4,5,6,7]
b[:-2]
b[:2]
b[-5:-3]
print(b[0:2])#左开右闭
a="abcdef"
c=a[:-3]
p=a.find("c")#返回c的位置
array=a.split("c")#分割，返回["ab","def"]
print(array)

#dict
m=dict()
m={}
m={"name":"xiaoming","age":13}
print(m["name"])

t=(1,2)#中括号是list,小括号是元组
#t[0]=123 不可以的，元组的元素是不能更改
t=list(t)
print(type(t))
t=(1,)#一个元素的tuple
print(type(t))#类型为元组
#如果将逗号去掉
t=(1)
print(type(t))#类型为int