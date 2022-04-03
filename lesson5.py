#一些冷门的操作
with map set lambda zip sort
#with
# 上下文管理器协议是用with 进行调用，默认调用了 enter 和 exit 两个魔法函数。
# 我们使用文件常常会遇到文件打开代码后没有关闭指令或者文件发生异常的问题，这时我们可以使用python中with语句，
# with 语句适用于对资源进行访问的场合，确保不管使用过程中是否发生异常都会执行必要的“清理”操作，释放资源。
with open("a.txt","w") as f:
    f.write("abc")

class Mywith:
    def __enter__(self):
        print("enter mywith")
        return self
    def __exit__(self,exc_type,exc_val,exc_tb):
        print("exit mywith")
with Mywith() as my:
    print("in with")
    print(my)
# 这里输出
# enter mywith
# in with
# exit mywith
# 在我们后面使用torch时候会用到
# with torch.no_grad():
    # pass
# 集合set
# 相当于没有值，只有key的字典

a=[1,1,2,3,5,5]
b=[3,5,9,10]
print(0 in a)
print(0 in set(a))
# 使用交集并集
print(set(a)&set(b))
print(set(a)|set(b))

# zip使用打包神器
a=[1,2,3]
b=["a","b","c"]
print((list(zip(a,b))))#[(1,'a'),(2,'b'),(3,'c')]

#map

#lambda

#sort
a=[1,3,2,0]
a.sort()
print(a)