# opencv

import cv2

image=cv2.imread("image.png") # H,W,C C,channel,3,BGR格式
print(image.shape,image.dtype)#(570,623,3) dtype('uint8')  就是numpy.ndarray

image=image[:,:,[2,1,0]]#改变通道顺序，符合正常图片,利用了numpy的选择功能实现BGR和RGB对调
print(image)
# cv2.imshow("1",image) cv2就是显示BGR，你用RGB反而不对呢

#转换成hsv
hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

#画框

# print(cv2.rectangle.__doc__)

x,y,r,b=150,100,450,480
cv2.rectangle(hsv,(x,y),(r,b),(0,255,0),2)#有的版本是需要返回image，然后再去使用image，有的版本不需要返回，直接就可以使用image
# cv2.imshow("image",image)

#画圆圈
# print(cv2.circle.__doc__)
x,y=230,270 #center
# cv2.circle(image,(x,y),16,(0,0,255),-1,lineType=16)

#保存图像
cv2.imwrite("image.change.png",image)

# PIL.Image 属于python的标准图片处理库
from PIL import Image
img=Image.open("image.png")#类型不是numpy矩阵，是一个PIL类对象，可以转换为numpy类型，自己查一下
img.resize((600,300))
print(image)

# 与opencv的互操作
cv_img=cv2.imread("image.png")
pil_img=Image.fromarray(cv_img)#将numpy格式的ndarray转换为PIL类型，这里也会存在BGR和RGB问题，在我们训练时一定要控制好
# 裁剪
crop=cv_img[150:450,100:480,:]#选择你想要的矩阵