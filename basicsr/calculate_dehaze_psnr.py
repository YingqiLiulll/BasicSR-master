import os
import numpy as np
import math
from PIL import Image

import time

tis =time.perf_counter()

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    # print("img1/1. - img2 / 1.",img1 / 1. - img2 / 1.)
    # print("img2 / 1.",img2 / 1.)
    # print("img2",img2)
    # mse = np.mean((img1 - img2) ** 2)
    print("mse:",mse)
    if mse < 1.0e-10:
        return 100 * 1.0
    return 20 * math.log10(255.0/ math.sqrt(mse))


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


path1 = 'C:/Users/13637/Desktop/sr'  # 指定输出结果文件夹
path2 = 'F:/dehaze/nyuhaze500_square/gt'  # 指定原图文件夹
f = os.listdir(path1)
# f_nums = len(os.listdir(path1))
def creatlist(r1,r2):
    return [item for item in range(r1,r2+1)]
r1, r2 =1400, 1404
name_list = creatlist(r1,r2)
name_list = [str(i) for i in name_list]
list_psnr = []
list_ssim = []
list_mse = []
for j in range(len(name_list)):
    for i in f:
        if name_list[j] in i:
            pf_a = os.path.join(path1, i)
            # print("pf_a:",pf_a)
            img_a = Image.open(pf_a).convert('RGB')
            img_a = img_a.resize((480,480))
            pf_b = os.path.join(path2, name_list[j]+'.png')
            # print("pf_b:",pf_b)
            img_b = Image.open(pf_b).convert('RGB')
            img_b = img_b.resize((480,480))
            # img_b.show()
            img_a = np.array(img_a).astype(np.uint8)
            img_b = np.array(img_b).astype(np.uint8)
            # print(img_a)
            # print(img_b[100:250,:,1])
            # img_b = Image.fromarray(img_b.astype('uint8')).convert('RGB')
            # img_b.show()
            psnr_num = psnr(img_a, img_b)
            print("psnr_num:",psnr_num)
            print(psnr_num)
            ssim_num = ssim(img_a, img_b)
            mse_num = mse(img_a, img_b)
            list_ssim.append(ssim_num)
            list_psnr.append(psnr_num)
            list_mse.append(mse_num)

print("平均PSNR:", np.mean(list_psnr))  # ,list_psnr)
print("平均SSIM:", np.mean(list_ssim))  # ,list_ssim)
print("平均MSE:", np.mean(list_mse))  # ,list_mse)

print("Time used:", tis)
