import cv2
import numpy as np
import scipy.misc
import glob

img = cv2.imread('F:/classical_SR_datasets/Set14/GTmod12/face.png')
print("img:",img.shape)

def modcrop(image, scale=12):
    #  image.shape is a tuple(元组)，len()以后是元组的维度
    if len(image.shape) == 3:
        h, w, _ = image.shape
        # 将h、wz转为12的倍数
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        # 得到裁剪后的图片
        image = image[0:h, 0:w, :]
    else:
        h, w = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w]
    # 返回值为裁剪后的图片
    return image

print(modcrop(img).shape)  # 裁剪后的尺寸


