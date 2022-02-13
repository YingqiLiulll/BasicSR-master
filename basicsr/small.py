import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image
import numpy as np
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-file', type=str, default='C:/Users/13637/Desktop/val_pic/2lossopt_01barbara_5000.png')
    args = parser.parse_args()
    image = mpimg.imread(args.image_file)
    fig, ax = plt.subplots(1, 1)
    plt.axis('off')
    axins =  inset_axes(ax, width="20%", height="20%", loc='lower left',
                   bbox_to_anchor=(0.63, 0, 1.8, 1.8),
                   bbox_transform=ax.transAxes)
    #image1=image[50:90,150:190,:]
    image1=image[100:200,450:550,:]
    axins.imshow(image1)
    #rect = plt.Rectangle((150, 50), 40, 40, fill=False, edgecolor='red', linewidth=3)
    rect = plt.Rectangle((450, 100), 100, 100, fill=False, edgecolor='red', linewidth=3)
    ax.add_patch(rect)
    #mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='r', lw=3)
    plt.axis('off')
    ax.imshow(image)

    savefig("C:/Users/13637/Desktop/plot.png".format(args.image_file),bbox_inches='tight')
    plt.show()
# testsets = "E:\pythonProject/0855.png"
# testset_H = f"E:\pythonProject/{args.image_file}"
# image = np.array(Image.open(testsets).convert('RGB'))
# image1 = np.array(Image.open(testset_H).convert('RGB'))
# pr = psnr(image, image1)
# print(pr)
