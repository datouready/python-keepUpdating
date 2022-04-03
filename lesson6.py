import numpy as np
import cv2
#数学运算，python数学运算库，比较出名
a=np.zeros((3,3),np.uint8)
a[:,0]=3#将第一列所有行变为3
print (a)
a=np.zeros((100,100,3),np.uint8)
a[:,:,1]=255
cv2.imshow("image",a)
cv2.waitKey()
