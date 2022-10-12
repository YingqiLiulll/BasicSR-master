import os
import glob
import cv2


def main():
    # folder = '/home/yqliu/projects/ClassSwin/data/DIV2K_scale_sub_psnr_LR_class2'
    folder = 'C:/Users/13637/Desktop/test/test_pic'
    DIV2K(folder)
    print('Finished.')


def DIV2K(path):
    index = 0
    img_path_l = os.listdir(path)
    for img_path in img_path_l:
        print(index)
        index += 1
        if ((len(os.path.splitext(img_path)[0])) > 7 and (len(os.path.splitext(img_path)[0])) < 10):
            # new_path = img_path.replace(os.path.basename(img_path), str(1.0) + "_" + os.path.basename(img_path))
            new_path = str(1.0) + "_" + os.path.basename(img_path)
            os.rename(os.path.join(path, img_path), os.path.join(path, new_path))

if __name__ == "__main__":
    main()