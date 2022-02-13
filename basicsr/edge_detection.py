import numpy as np
import cv2
from PIL import Image
import imageio

def img_show(img):
    cv2.namedWindow("Image")
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def sub_plot(img_1, img_2):
    l = np.size(img, 1) / 4  # a quarter of the columns
    rows = np.size(img, 0)
    interval = np.ones((rows, int(l)))
    interval = interval * 255

    img_o = np.concatenate((img_1, interval, img_2), axis=1)
    return img_o


def sobel_v(img, threshold):
    '''
    edge detection with the vertical Sobel filter

    Parameters
    ----------
    img : TYPE
        the image input.
    threshold : TYPE
         varies for application [0 255].

    Returns
    -------
    mag : TYPE
        output after edge detection.

    '''
    G_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    rows = np.size(img, 0)
    columns = np.size(img, 1)
    mag = np.zeros(img.shape)
    for i in range(0, rows - 2):
        for j in range(0, columns - 2):
            v = sum(sum(G_x * img[i:i + 3, j:j + 3]))  # vertical
            mag[i + 1, j + 1] = v

    for p in range(0, rows):
        for q in range(0, columns):
            if mag[p, q] < threshold:
                mag[p, q] = 0
    return mag


def sobel_h(img, threshold):
    '''
    edge detection with the horizon Sobel filter

    Parameters
    ----------
    img : TYPE
        the image input.
    threshold : TYPE
         varies for application [0 255].

    Returns
    -------
    mag : TYPE
        output after edge detection.

    '''
    G_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    rows = np.size(img, 0)
    columns = np.size(img, 1)
    mag = np.zeros(img.shape)
    for i in range(0, rows - 2):
        for j in range(0, columns - 2):
            h = sum(sum(G_y * img[i:i + 3, j:j + 3]))  # horizon
            mag[i + 1, j + 1] = h

    for p in range(0, rows):
        for q in range(0, columns):
            if mag[p, q] < threshold:
                mag[p, q] = 0
    return mag


def sobel(img, threshold):
    '''
    edge detection based on sobel

    Parameters
    ----------
    img : TYPE
        the image input.
    threshold : TYPE
         varies for application [0 255].

    Returns
    -------
    mag : TYPE
        output after edge detection.

    '''
    G_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    G_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    rows = np.size(img, 0)
    columns = np.size(img, 1)
    mag = np.zeros(img.shape)
    for i in range(0, rows - 2):
        for j in range(0, columns - 2):
            v = sum(sum(G_x * img[i:i + 3, j:j + 3]))  # vertical
            h = sum(sum(G_y * img[i:i + 3, j:j + 3]))  # horizon
            mag[i + 1, j + 1] = np.sqrt((v ** 2) + (h ** 2))

    for p in range(0, rows):
        for q in range(0, columns):
            if mag[p, q] < threshold:
                mag[p, q] = 0
    return mag

img = cv2.imread('F:/classical_SR_datasets/Set14/GTmod12/foreman.png', 0)  # read an image
mag = sobel(img, 70)
mag_v = sobel_v(img, 70)
mag_h = sobel_h(img, 70)
scipy.misc.imsave('C:/Users/13637/Desktop/foreman_edge.png', mag)
im = Image.fromarray(mag)
im = im.convert("L")
im.save("C:/Users/13637/Desktop/val_pic/foreman_edge.png")

img_show(mag)

v = sub_plot(img, mag_v)
h = sub_plot(img, mag_h)
a = sub_plot(img, mag)

