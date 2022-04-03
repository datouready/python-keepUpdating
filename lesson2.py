#逻辑语
a=True


if a:
    print("abc")#使用table来控制逻辑顺序


while a:
    print("abc")
    break
# 类型支持iter
# iter=list tuple 支持iter的类
# for var in iter:
#range(10)生成0-10的lsit
for i in range(10):
    print(i)

#想知道索引index
var=[3,2,5]
for index,value in enumerate(var):
    print(f"{index},{value}")

for value in enumerate(var):
    print(f"{value}")#这里得到的是tuple

# for循环的一点高级写法
a=[item*2 for item in var]#意思生成一个list，从var里面取出来，然后+2
a=[item*2 for item in var if item > 2]#意思生成一个list，从var里面取出来，然后+2 

#c++ a=isok ? 1:2
isok=True
a=1 if isok else 2
if isok:
    a=1
elif a==2:
    a=5
else:
    a=2

#取代&&  ||
if isok or a==5:

if isok and a==5: