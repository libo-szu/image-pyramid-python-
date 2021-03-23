import cv2
import numpy as np


import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from Convelution  import Conv,Padding
import matplotlib.pyplot  as plt
class  Img_pyramid(object):
    def __init__(self):
        self.down_gaussian_kernel=1/256 * np.array(
        [
            [1,  4,  6,  4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1,  4,  6,  4, 1]
        ])
        self.up_gaussian_kernel=4/256 * np.array(
        [
            [1,  4,  6,  4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1,  4,  6,  4, 1]
        ])
    def gaussian_pyramid(self,img,num_layer):
        result_pyramid=[img]
        for i in range(num_layer):
            img=self.down_sample(img)
            result_pyramid.append(img)
        return result_pyramid
    def lapalian_pyramid(self,img,num_layer):
        result_pyramid=[]
        gaussian_pyramid=self.gaussian_pyramid(img,num_layer) #6
        result_pyramid.append(gaussian_pyramid[-1])
        for i in range(len(gaussian_pyramid)-1,0,-1):
            low_img=gaussian_pyramid[i]
            low_img=self.up_sample(low_img).astype("float32")
            high_img=gaussian_pyramid[i-1].astype("float32")

            lapalian_img=cv2.subtract(high_img,low_img)
            result_pyramid.append(lapalian_img)
            plt.imshow(lapalian_img)
            plt.show()
        return result_pyramid



    def up_sample(self,img):
        w,h=img.shape
        for i in range(w):
            img=np.insert(img,i*2+1,0,axis=0)
        for j in range(h):
            img=np.insert(img,j*2+1,0,axis=1)
        img=Conv(kernel=self.up_gaussian_kernel,padding="constant")(img)
        return img
    def down_sample(self,img):
        img=Conv(kernel=self.down_gaussian_kernel,padding="constant")(img)
        img=img[::2,::2]
        return img


if __name__=="__main__":
    img=cv2.imread("1.jpg",0)
    img=cv2.resize(img,(256,256))
    a=Img_pyramid()
    img_gaussian_pyramid=a.gaussian_pyramid(img,3)
    img_lapalian_pyramid=a.lapalian_pyramid(img,3)


