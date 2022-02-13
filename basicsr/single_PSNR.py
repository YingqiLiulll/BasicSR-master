from PIL import Image
import numpy
import math


def psnr(img1, img2):
    mse = numpy.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


img1 = Image.open('C:/Users/13637/Desktop/Set5BIL/x4BIL/baby.png')
img2 = Image.open('C:/Users/13637/Desktop/Set5BIL/GTmod12/baby.png')

i1_array = numpy.array(img1)
i2_array = numpy.array(img2)

r12 = psnr(i1_array, i2_array)
print("PSNR:",r12)