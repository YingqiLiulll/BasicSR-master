from PIL import Image
import numpy
import math


def psnr(img1, img2):
    mse = numpy.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    print(img1 - img2)
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    # return 10 * math.log10(PIXEL_MAX**2 / mse)

def psnr2(img1, img2):
   mse = numpy.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

img1 = Image.open('C:/Users/13637/Desktop/1449_9_5000.png')
img2 = Image.open('F:/dehaze/nyuhaze500/gt/1449.png')
scale = 1

img2 = img2.crop([0,0,img1.size[0] * scale,img1.size[1] * scale])
i1_array2 = numpy.array(img1,dtype="float32")
i2_array2 = numpy.array(img2,dtype="float32")
# print(i1_array.shape)
i1_array = numpy.array(img1)
i2_array = numpy.array(img2)
# i1_array = i1_array.astype(numpy.float64)
# i2_array = i2_array.astype(numpy.float64)
print("i1_array2",i1_array2)
print("i2_array2",i2_array2)
# print("i1_array.dtype",i1_array.dtype)
# print("i2_array.dtype",i2_array.dtype)

r12 = psnr(i1_array, i2_array)
r12V2 = psnr2(i1_array, i2_array)
r12_float = psnr(i1_array2, i2_array2)
print("PSNR:",r12, r12V2, r12_float)