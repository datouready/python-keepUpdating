import os
import sys
print(os.listdir("."))#查看当前路径下文件
#基本数据类型
#int,float,str,list,dict,set,tuple
a=123
a=int("123")
b=123.
b=float("123")
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