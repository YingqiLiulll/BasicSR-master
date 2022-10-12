import os
import glob
import cv2


def main():
    # folder = '/home/yqliu/projects/ClassSwin/data/DIV2K_scale_sub_psnr_LR_class2'
    folder = 'C:/Users/13637/Desktop/test/test_pic'
    # save_folder = '/home/yqliu/projects/ClassSwin/data/DIV2K_scale_sub_psnr_LR_class2_rename'
    save_folder = 'C:/Users/13637/Desktop/test/new'
    for i in [save_folder]:
        if os.path.exists(i):
            pass
        else:
            os.makedirs(i)
    DIV2K(folder, save_folder)
    print('Finished.')

def DIV2K(path, save_folder):
    index = 0
    img_name_l = os.listdir(path)
    for img_name in img_name_l:
        print(index)
        index += 1
        img_path = os.path.join(path, img_name)
        IMG = cv2.imread(img_path)
        if ((len(os.path.splitext(img_name)[0])) > 7 and (len(os.path.splitext(img_name)[0])) < 10):
            cv2.imwrite(os.path.join(save_folder, str(1.0) + "_" + os.path.basename(img_name)), IMG)
            new_name = str(1.0) + "_" + os.path.basename(img_name)
            print(os.path.basename(new_name))
        else:
            cv2.imwrite(os.path.join(save_folder, os.path.basename(img_name)), IMG)
            print(os.path.basename(img_name))

if __name__ == "__main__":
    main()