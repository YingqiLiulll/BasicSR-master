from PIL import Image
import numpy
import math


def psnr(img1, img2):
    mse = numpy.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    # return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return 10 * math.log10(PIXEL_MAX**2 / mse)


img1 = Image.open('F:/handyview_pic/images/1400_8_500002.png')
img2 = Image.open('F:/handyview_pic/images/1400.png')
scale = 1

img2 = img2.crop([0,0,img1.size[0] * scale,img1.size[1] * scale])
i1_array = numpy.array(img1)
i2_array = numpy.array(img2)
# print(i1_array)
# print(i2_array)

r12 = psnr(i1_array, i2_array)
print("PSNR:",r12)