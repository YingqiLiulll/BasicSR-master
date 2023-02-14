import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from PIL import Image
from io import BytesIO
from basicsr.data.degradations import random_bivariate_Gaussian
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from torch.utils import data as data


@DATASET_REGISTRY.register()
class PretrainDataset(data.Dataset):
    """Dataset used for Pretrain model:
    data process goal: divide a 128x128 image into 8 16x16 grids, every grid is given a kind of random degradation.

    It loads gt (Ground-Truth) images, and augments them.
    It also generates blur kernels / noise and jpeg compression for generating low-quality images.
    Note that the low-quality images are processed in tensors on GPUS for faster processing.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            Please see more options in the codes.
    """
    def __init__(self, opt):
        super(PretrainDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.gt_folder = opt['dataroot_gt']

        # file client (lmdb io backend)
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            if not self.gt_folder.endswith('.lmdb'):
                raise ValueError(f"'dataroot_gt' should end with '.lmdb', but received {self.gt_folder}")
            with open(osp.join(self.gt_folder, 'meta_info.txt')) as fin:
                self.paths = [line.split('.')[0] for line in fin]
        else:
            # disk backend with meta_info
            # Each line in the meta_info describes the relative path to an image
            with open(self.opt['meta_info']) as fin:
                paths = [line.strip().split(' ')[0] for line in fin]
                self.paths = [os.path.join(self.gt_folder, v) for v in paths]

        # -------------------------------- the degradation settings -------------------------------- #

        # blur settings for degradation
        self.kernel_range = [2 * v + 1 for v in range(1, 4)]  # kernel size ranges from 7 to 21
        self.blur_sigma = opt['blur_sigma']
    


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.paths[index]
        # avoid errors caused by high latency in reading files
        retry = 3
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path, 'gt')
            except (IOError, OSError) as e:
                logger = get_root_logger()
                logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__())
                gt_path = self.paths[index]
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1
        img_gt = imfrombytes(img_bytes, float32=True)
        # img_gt is numpy

        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        img_gt = augment(img_gt, self.opt['use_hflip'], self.opt['use_rot'])

        # crop or pad to 128
        # TODO: 128 is hard-coded. You may change it accordingly
        h, w = img_gt.shape[0:2]
        crop_pad_size = 128
        # pad
        if h < crop_pad_size or w < crop_pad_size:
            pad_h = max(0, crop_pad_size - h)
            pad_w = max(0, crop_pad_size - w)
            img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        # crop
        if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
            h, w = img_gt.shape[0:2]
            # randomly choose top and left coordinates
            top = random.randint(0, h - crop_pad_size)
            left = random.randint(0, w - crop_pad_size)
            img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]

        # ------------------------ Generate kernels (used in the degradation) ------------------------ #
        # degradation = random.choice(self.degradation_list)
        # if needed, could randomly select one type of degradation one time, repeat this option for eight times

        kernel_size = random.choice(self.kernel_range)
        kernel = random_bivariate_Gaussian(
                        kernel_size, sigma_x_range=self.blur_sigma, sigma_y_range=self.blur_sigma, rotation_range=[-math.pi, math.pi], noise_range=None, isotropic=True)
        # pad kernel
        pad_size = (7 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

    # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        kernel = torch.FloatTensor(kernel)
        return_d = {'gt': img_gt, 'kernel': kernel, 'gt_path': gt_path}
        return return_d

    def __len__(self):
        return len(self.paths)