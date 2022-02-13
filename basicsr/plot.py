import argparse
import matplotlib.image as mpimg
import PIL.Image as pil_image
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image-file', type=str, default='C:/Users/13637/Desktop/val_pic/2lossopt_02barbara_245000.png')

    args = parser.parse_args()
    image = pil_image.open(args.image_file).convert('RGB')
    image = image.crop((0, 0, 720, 576))

    # image = image.crop((5, 5, 250, 250))

    fig, ax = plt.subplots(1, 1)
    plt.axis("off")
    axins = inset_axes(ax, width="20%", height="20%", loc="lower left",
                       bbox_to_anchor=(0.6,0.1, 1.8, 1.8),
                       bbox_transform=ax.transAxes)
    image1 = image[50:90,40:80,:]
    plt.axis('off')
    axins.imshow(image1)

    rect = plt.Rectangle((40,50), 40, 40, fill=False, edgecolor='red', linewidth=3)
    # rect2 = plt.Rectangle((95, 95), 149, 149, fill=False, edgecolor='white', linewidth=5)

    ax.add_patch(rect)

    # ax.add_patch(rect2)

    plt.axis('off')
    ax.imshow(image)

    plt.show()