#函数
a=123#全局变量

def func():
    global a #想要修改全局变量，不然只可以使用，不能改变
    a=555
def func1():
    pass#不想实现就直接使用pass就可以了

def func2(a,b,c):
    print(f"a={a},b={b},c={c}")
# func2(a=1,b=2,c=3)

# 星号使用
# 一个*后面一定是接着list或者tuple,然后展开给参数
func2(*[1,2,3])
# 两个参数一定是字典dict,根据key去给参数赋值
func2(**{"a":1,"b":2,"c":3})

def func4(w):
    print("w={}".format(w))

def func3(a,b,*args0,**args1):
    print(f"args0={args0}")#args0=(33,55)这是tuple
    print(f"args1={args1}")#args1={'w':333}
    func4(**args1)#可以继续传递给func4使用

func3(1,2,33,55,w=333)

a,b,c=1,2,3
def func5():
    return 1,2
a,b=func5()#这里叫做拆包