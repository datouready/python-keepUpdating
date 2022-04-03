#  class

class Person(object):#可以继承object
    def __init__(self,name):
        print("构造了:{}".format(name))
        self.name=name #不用事先写
    # 定义自己的函数
    def call(self):
        print("person:{},call".format(self.name))
    def __len__(self):#这种带下划线的都是系统内置的特殊函数
        return 123
    def __getitem__(self,item):
        return "abc:"+item
person = Person(name="xiaoming")
#还可以无中生有
person.age=123
print(person.age)
person.call()
print(person["getItemCall"])
print(len(person))

print(__name__)#可以看是不是当前模块,是的话会输出__main__
if __name__=="__main__":
    print("startup)
