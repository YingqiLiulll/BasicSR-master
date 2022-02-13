import cv2
import os.path
import glob
import numpy as np

def modcrop(pngfile,outdir):
    image = cv2.imread(pngfile, cv2.IMREAD_ANYCOLOR)
    scale = 12
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

    cv2.imwrite(os.path.join(outdir, os.path.basename(pngfile)), image)


for pngfile in glob.glob(r'F:\classical_SR_datasets\manga109\*.png'):
    modcrop(pngfile, r'F:\data_val\Manga109\GTmod12')