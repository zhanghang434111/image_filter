import numpy as np
import matplotlib.pylab as plt
import matplotlib.image as img
from scipy.signal import convolve2d  #二维卷积
import imageFilter as imgfilter

# 加载图像
im = plt.imread("./image/girl.bmp") # 加载当前文件夹中名为girl.jpg的图片
print(im.shape) # 输出图像尺寸
# 这里用的是RGB三通道图像，通道数为3
plt.imshow(im)
plt.show()      #输出图像


#newimg = imgfilter.average_filter(im)
newimg = imgfilter.average_filter(im)
img.imsave('newgirl.bmp', newimg)
plt.imshow(newimg)
plt.show()


