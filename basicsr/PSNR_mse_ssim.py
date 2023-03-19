import os
import numpy as np
import math
from PIL import Image

import time

tis =time.perf_counter()

def psnr(img1, img2):
    mse = np.mean((img1 / 1. - img2 / 1.) ** 2)
    if mse < 1.0e-10:
        return 100 * 1.0
    return 10 * math.log10(255.0 * 255.0 / mse)


def mse(img1, img2):
    mse = np.mean((img1 / 1. - img2 / 1.) ** 2)
    return mse


def ssim(y_true, y_pred):
    u_true = np.mean(y_true)
    u_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    std_true = np.sqrt(var_true)
    std_pred = np.sqrt(var_pred)
    c1 = np.square(0.01 * 7)
    c2 = np.square(0.03 * 7)
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    return ssim / denom


path1 = 'F:/data_val_degra/Urban100/GTmod12'  # 指定输出结果文件夹
path2 = 'F:/data_val_degra/Urban100/LR_degra_x1/blur_noise_jpeg'  # 指定原图文件夹
f = os.listdir('F:/data_val_degra/Urban100/LR_degra_x1/blur_noise_jpeg')
# f_nums = len(os.listdir(path1))
list_psnr = []
list_ssim = []
list_mse = []
for i in f:
    pf_a = os.path.join('F:/data_val_degra/Urban100/LR_degra_x1/blur_noise_jpeg', i)
    img_a = Image.open(pf_a).convert('RGB')
    pf_b = os.path.join('F:/data_val_degra/Urban100/GTmod12', i)
    img_b = Image.open(pf_b).convert('RGB')
    img_a = np.array(img_a,dtype="float32")
    img_b = np.array(img_b,dtype="float32")

    psnr_num = psnr(img_a, img_b)
    ssim_num = ssim(img_a, img_b)
    mse_num = mse(img_a, img_b)
    list_ssim.append(ssim_num)
    list_psnr.append(psnr_num)
    list_mse.append(mse_num)
print("平均PSNR:", np.mean(list_psnr))  # ,list_psnr)
print("平均SSIM:", np.mean(list_ssim))  # ,list_ssim)
print("平均MSE:", np.mean(list_mse))  # ,list_mse)

print("Time used:", tis)
