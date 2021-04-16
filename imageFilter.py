import numpy as np
import matplotlib.pylab as plt
import matplotlib.image as img
from scipy.signal import convolve2d  #二维卷积
from scipy.ndimage import median_filter
import math

def convolve_myself(img,window) :
    kernel_size = window.shape[0]
    half_kernel = int((kernel_size-1)/2)
    height = img.shape[0]
    width = img.shape[1]
    # padding
    # 考虑边界问题，四周补0
    imgPadding = np.zeros((height + (kernel_size-1), width + (kernel_size-1)), dtype=np.uint8)
    imgPadding[half_kernel: height + half_kernel , half_kernel: width+ half_kernel ] = img[: , : ]

    result = np.zeros((height, width), dtype=np.uint8)
    inputs = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
    for row in range(height):
        for col in range(width):
            inputs = imgPadding[row : row + kernel_size, col : col + kernel_size]
            #print(inputs)
            # 积分
            outputs = inputs * window
            result[row,col] = np.sum(outputs)
            #print(result[row,col])
    return result

#对所有通道做卷积
def convolve_all_colours(im, window):
    """
    卷积图像，依次对图像的每个通道卷积
    """
    im_temp = []
    # 用ims作为每个通道转换结果的暂存列表
    for d in range(3):
    # 对图像的三个通道循环处理
        im_conv_d = convolve2d(im[:,:,d], window, mode="same", boundary="symm")
        # mode决定输出尺寸，boundary决定边界条件，这里输出尺寸与原图相同，采用对称边界条件
        im_temp.append(im_conv_d)
        # 将单通道转换结果添加到列表
    im_conv = np.stack(im_temp, axis=2).astype("uint8")
    # 在第三维上堆叠ims列表中的每个元素，并通过astype保证值在0-255
    return im_conv

def average_filter(img,kernel_size=3):
    n = kernel_size
    average_window = np.ones((n, n))
    # 构建7x7的全1矩阵（均值滤波）
    average_window /= np.sum(average_window)
    # 矩阵每个元素除以矩阵所有元素的和，使矩阵所有元素的和为1
    new_img = convolve_all_colours(img, average_window)
    return new_img

def median_filter_myself(img, kernel_size):
    """
    对图像所有通道运用中值滤波
    """
    im_temp = []
    for d in range(3):
        im_conv_d = median_filter(img[:, :, d], size=(kernel_size, kernel_size))
        im_temp.append(im_conv_d)
    im_conv = np.stack(im_temp, axis=2).astype("uint8")
    return im_conv


# 高斯滤波模板
def guassian_window(n, sigma=1):
    """
    使用高斯分布的权重创建一个n*n的方形窗口
    """
    nn = int((n - 1) / 2)
    a = np.asarray([[x ** 2 + y ** 2 for x in range(-nn, nn + 1)] for y in range(-nn, nn + 1)])
    # np.asarray可以将输入转化为np.array, 这里输入为一个列表推导式
    b = np.exp(-a / (2 * sigma ** 2)) # / (2 * math.pi * sigma ** 2)
    # 省略掉常数项 1 / (2 * math.pi * sigma ** 2)  不会影响互相之间的比例关系，最终都要进行归一化

    # 权重归一化
    b = b / np.sum(b)
    return b

def gussian_filter(img,kernel_size=3,sigma=1):
    g_window = guassian_window(kernel_size, sigma=sigma)
    new_img = convolve_all_colours(img, g_window)
    return new_img


# 自实现的图像双边滤波
# -------------------------------------------------------------
def bilateral_filter(img, kernel=3, sigmaSpace=10, sigmaColor = 100, padding="VALID"):
    row = img.shape[0]
    col = img.shape[1]
    channels = img.shape[2]
    result = img.copy()
    half_kernel = int((kernel - 1) / 2)
    for c in range(channels):
        for i in range(half_kernel,row-half_kernel-1):  # 对每一个点进行处理
            for j in range(half_kernel,col-half_kernel-1):
                weightSum = 0
                filterValue = 0
                for row_d in range(-half_kernel, half_kernel):
                    for col_d in range(-half_kernel, half_kernel):
                        distance_Square = row_d * row_d + col_d * col_d
                        position_x = i + half_kernel + row_d
                        position_y = j + half_kernel + col_d
                        value_Square = np.power(img[i, j, c] - img[position_x, position_y, c], 2)
                        weight = np.exp(-1 * (distance_Square / (2 * sigmaSpace ** 2) + value_Square / (
                                    2 * sigmaColor ** 2)))
                        weightSum += weight  # 权重和归一化
                        filterValue += (weight * img[position_x, position_y, c])
                outputs = filterValue / weightSum
                result[i, j, c] = outputs

    return result







