import os
import glob
import cv2
from pathlib import Path


def main():
    # folder = '/home/yqliu/projects/ClassSwin/data/DIV2K_scale_sub_psnr_LR_class2'
    folder = 'F:/Real-Test/Challenge/Challenge/Canon/Pair_test/HR'
    DIV2K(folder)
    print('Finished.')


def DIV2K(path):
    index = 0
    img_path_l = os.listdir(path)
    for img_path in img_path_l:
        print(index)
        index += 1
        # print("img_path:",img_path)
        basename, ext = os.path.splitext(img_path)
        new_basename = basename.strip(' ')
        # parts = basename.split('-')
        # if ' 3' in basename:
        #     new_basename = basename.replace(' 3',' ').strip(' ')
        # elif '_HR' in basename:
        #     new_basename = basename.replace('_HR',' ').strip('')
        new_path = new_basename + ext
        os.rename(os.path.join(path, img_path), os.path.join(path, new_path))
        # print("basename, ext:",basename, ext)
        # if ((len(os.path.splitext(img_path)[0])) > 7 and (len(os.path.splitext(img_path)[0])) < 10):
        #     # new_path = img_path.replace(os.path.basename(img_path), str(1.0) + "_" + os.path.basename(img_path))
        #     new_path = str(1.0) + "_" + os.path.basename(img_path)
        #     os.rename(os.path.join(path, img_path), os.path.join(path, new_path))

if __name__ == "__main__":
    main()