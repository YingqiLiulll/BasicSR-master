import os.path
import torch
from PIL import Image
import torchvision.transforms as standard_transforms
import numpy as np
import random

import math
import cv2
import numpy as np
import os.path as osp
import os
from basicsr.utils import DiffJPEG
from basicsr.utils.img_process_util import filter2D
from basicsr.data.degradations import random_bivariate_Gaussian
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt

def check_dir(dir):
    if os.path.exists(dir):
        pass
    else:
        os.makedirs(dir)



if __name__ == '__main__':

    # for i in ['Set5','Set14','BSD100','Manga109','Urban100']:
    for i in ['Urban100']:

        GT_folder = 'F:/data_val_degra/{}/GTmod12'.format(i)
        save_LR_folder = 'F:/data_val_degra/{}/LR_degra_x2'.format(i)

        transform = standard_transforms.ToTensor()
        Retransform = standard_transforms.ToPILImage()
        img_GT_list = sorted(os.listdir(GT_folder))

        for path_GT in img_GT_list:
            print(path_GT)

            img_GT = Image.open(os.path.join(GT_folder,path_GT))
            img_GT_tensor = transform(img_GT).unsqueeze(0)

            # only blur
            img_clone = img_GT_tensor.clone()
            kernel_range = [2 * v + 1 for v in range(1, 4)]  # kernel size ranges from 7 to 21
            blur_sigma = [0.2, 3]
            # ------------------------ Generate kernels (used in the degradation) ------------------------ #
            kernel_size = random.choice(kernel_range)
            kernel = random_bivariate_Gaussian(
                        kernel_size, sigma_x_range=blur_sigma, sigma_y_range=blur_sigma, rotation_range=[-math.pi, math.pi], noise_range=None, isotropic=True)
            # pad kernel
            pad_size = (7 - kernel_size) // 2
            kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
            kernel = torch.FloatTensor(kernel)
            img_LR = filter2D(img_clone, kernel)
            img_LR = Retransform(img_LR.squeeze(0))
            save_dir = os.path.join(save_LR_folder,'blur')
            check_dir(save_dir)
            img_LR.save(os.path.join(save_dir,os.path.basename(path_GT)))

            #only noise
            gray_noise_prob = 0.5
            noise_range = [1, 30]
            poisson_scale_range = [0.05, 3]
            img_clone = img_GT_tensor.clone()
            if np.random.uniform() < gray_noise_prob:
                img_LR = random_add_gaussian_noise_pt(img_clone, sigma_range=noise_range, clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                img_LR = random_add_poisson_noise_pt(
                    img_clone,
                    scale_range = poisson_scale_range,
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
            img_LR = Retransform(img_LR.squeeze(0))
            save_dir = os.path.join(save_LR_folder, 'noise')
            check_dir(save_dir)
            img_LR.save(os.path.join(save_dir,os.path.basename(path_GT)))

            # only jepg
            jpeger = DiffJPEG(differentiable=False)  # simulate JPEG compression artifacts
            jpeg_range = [30, 95]

            img_clone = img_GT_tensor.clone()
            jpeg_p = img_clone.new_zeros(img_clone.size(0)).uniform_(*jpeg_range)
            out = torch.clamp(img_clone, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant
            out = jpeger(out, quality=jpeg_p)
            img_LR = torch.clamp((out * 255.0).round(), 0, 255) / 255.
            img_LR = Retransform(img_LR.squeeze(0))
            save_dir = os.path.join(save_LR_folder, 'jepg')
            check_dir(save_dir)
            img_LR.save(os.path.join(save_dir,os.path.basename(path_GT)))

            # blur+noise
            img_clone = img_GT_tensor.clone()
            img_blur = filter2D(img_clone, kernel)
            if np.random.uniform() < gray_noise_prob:
                img_LR = random_add_gaussian_noise_pt(img_blur, sigma_range=noise_range, clip=True, rounds=False, gray_prob = gray_noise_prob)
            else:
                img_LR = random_add_poisson_noise_pt(
                    img_blur,
                    scale_range = poisson_scale_range,
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
            img_LR = Retransform(img_LR.squeeze(0))
            save_dir = os.path.join(save_LR_folder, 'blur_noise')
            check_dir(save_dir)
            img_LR.save(os.path.join(save_dir,os.path.basename(path_GT)))


            # blur+jpeg
            # blur
            img_clone = img_GT_tensor.clone()
            img_blur = filter2D(img_clone, kernel)
            # jpeg
            jpeg_p = img_blur.new_zeros(img_blur.size(0)).uniform_(*jpeg_range)
            out = torch.clamp(img_blur, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant
            out = jpeger(out, quality=jpeg_p)
            img_LR = torch.clamp((out * 255.0).round(), 0, 255) / 255.
            img_LR = Retransform(img_LR.squeeze(0))
            save_dir = os.path.join(save_LR_folder, 'blur_jpeg')
            check_dir(save_dir)
            img_LR.save(os.path.join(save_dir, os.path.basename(path_GT)))

            # noise+jpeg
            # noise
            img_clone = img_GT_tensor.clone()
            if np.random.uniform() < gray_noise_prob:
                img_noise = random_add_gaussian_noise_pt(img_clone, sigma_range=noise_range, clip=True, rounds=False,   gray_prob = gray_noise_prob)
            else:
                img_noise = random_add_poisson_noise_pt(
                    img_clone,
                    scale_range = poisson_scale_range,
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
            # jpeg
            jpeg_p = img_noise.new_zeros(img_noise.size(0)).uniform_(*jpeg_range)
            out = torch.clamp(img_noise, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant
            out = jpeger(out, quality=jpeg_p)
            img_LR = torch.clamp((out * 255.0).round(), 0, 255) / 255.
            img_LR = Retransform(img_LR.squeeze(0))
            save_dir = os.path.join(save_LR_folder, 'noise_jpeg')
            check_dir(save_dir)
            img_LR.save(os.path.join(save_dir, os.path.basename(path_GT)))


            # blur+noise+jpeg
            # blur
            img_clone = img_GT_tensor.clone()
            img_blur = filter2D(img_clone, kernel)
            # noise
            if np.random.uniform() < gray_noise_prob:
                img_blur_noise = random_add_gaussian_noise_pt(img_blur, sigma_range=noise_range, clip=True, rounds=False, gray_prob = gray_noise_prob)
            else:
                img_blur_noise = random_add_poisson_noise_pt(
                    img_blur,
                    scale_range = poisson_scale_range,
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
            # jpeg
            jpeg_p = img_blur_noise.new_zeros(img_blur_noise.size(0)).uniform_(*jpeg_range)
            out = torch.clamp(img_blur_noise, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant
            out = jpeger(out, quality=jpeg_p)
            img_LR = torch.clamp((out * 255.0).round(), 0, 255) / 255.
            img_LR = Retransform(img_LR.squeeze(0))
            save_dir = os.path.join(save_LR_folder, 'blur_noise_jpeg')
            check_dir(save_dir)
            img_LR.save(os.path.join(save_dir, os.path.basename(path_GT)))