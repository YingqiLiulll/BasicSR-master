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
from basicsr.utils import FileClient, bgr2ycbcr, get_root_logger, imfrombytes, img2tensor, scandir
from basicsr.utils.registry import DATASET_REGISTRY
from torch.utils import data as data


@DATASET_REGISTRY.register()
class PretrainV2Dataset(data.Dataset):
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
        super(PretrainV2Dataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.gt_folder = opt['dataroot_gt']

        # file client (lmdb io backend)
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
        elif 'meta_info' in self.opt:
            with open(self.opt['meta_info'], 'r') as fin:
                self.paths = [osp.join(self.gt_folder, line.split(' ')[0]) for line in fin]
        else:
            self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))


        # -------------------------------- the degradation settings -------------------------------- #

        # blur settings for degradation
        self.kernel_range = [2 * v + 1 for v in range(1, 4)]  # kernel size ranges from 7 to 21
        self.blur_sigma = opt['blur_sigma']

    def random_crop(self, img_gts, gt_patch_size, gt_path=None):
        """Random crop. Support Numpy array and Tensor inputs.

        It crops lists of gt images.

        Args:
            img_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT images. Note that all images
                should have the same shape. If the input is an ndarray, it will
                be transformed to a list containing itself.
            gt_patch_size (int): GT patch size.
            gt_path (str): Path to ground-truth. Default: None.

        Returns:
            list[ndarray] | ndarray: GT images. If returned results
                only have one element, just return ndarray.
        """
        if not isinstance(img_gts, list):
            img_gts = [img_gts]
        # determine input type: Numpy array or Tensor
        input_type = 'Tensor' if torch.is_tensor(img_gts[0]) else 'Numpy'
        if input_type == 'Tensor':
            h_gt, w_gt = img_gts[0].size()[-2:]
        else:
            h_gt, w_gt = img_gts[0].shape[0:2]
        if h_gt < gt_patch_size or w_gt < gt_patch_size:
            raise ValueError(f'LQ ({h_gt}, {w_gt}) is smaller than patch size '
                            f'({gt_patch_size}, {gt_patch_size}). '
                            f'Please remove {gt_path}.')
        # randomly choose top and left coordinates for gt patch
        top = random.randint(0, h_gt - gt_patch_size)
        left = random.randint(0, w_gt - gt_patch_size)

        # crop gt patch
        if input_type == 'Tensor':
            img_gts = [v[:, :, top:top + gt_patch_size, left:left + gt_patch_size] for v in img_gts]
        else:
            img_gts = [v[top:top + gt_patch_size, left:left + gt_patch_size, ...] for v in img_gts]
        if len(img_gts) == 1:
            img_gts = img_gts[0]
        return img_gts


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.paths[index]
        # img_bytes = self.file_client.get(gt_path, 'gt')
        
        # wait to be solved, you can uncomment these for more advanced training
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

        # -------------- Do augmentation for training: random_crop, flip, rotation -------------------- #
        gt_size = self.opt['gt_size']
        img_gt = self.random_crop(img_gt, gt_size, gt_path)
        img_gt = augment(img_gt, self.opt['use_hflip'], self.opt['use_rot'])

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]

        # crop or pad to gt_size
        h, w = img_gt.shape[0:2]
        crop_pad_size = self.opt['gt_size']
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
