
import random
import numpy as np
import cv2
import lmdb
import torch
import basicsr.data.util as util
from basicsr.data.transforms import augment, paired_random_crop
from torch.utils import data as data
import os
import math
from basicsr.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class LQGTDataset(data.Dataset):
    '''
    Read LQ (Low Quality, here is LR) and GT image pairs.
    If only GT image is provided, generate LQ image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, opt):
        super(LQGTDataset, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.paths_lq, self.paths_gt = None, None
        self.sizes_lq, self.sizes_gt = None, None
        self.lq_env, self.gt_env = None, None  # environments for lmdb

        self.paths_gt, self.sizes_gt = util.get_image_paths(self.data_type, opt['dataroot_gt'])
        self.paths_lq, self.sizes_lq = util.get_image_paths(self.data_type, opt['dataroot_lq'])

        # self.paths_GT=[self.paths_GT[400000],self.paths_GT[800000],self.paths_GT[1200000]]
        # self.paths_LQ=[self.paths_LQ[400000],self.paths_LQ[800000],self.paths_LQ[1200000]]


        assert self.paths_gt, 'Error: GT path is empty.'
        # if self.paths_LQ and self.paths_GT:
        #     assert len(self.paths_LQ) == len(
        #         self.paths_GT
        #     ), 'GT and LQ datasets have different number of images - {}, {}.'.format(
        #         len(self.paths_LQ), len(self.paths_GT))
        self.random_scale_list = [1]

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.gt_env = lmdb.open(self.opt['dataroot_gt'], readonly=True, lock=False, readahead=False,
                                meminit=False)
        self.lq_env = lmdb.open(self.opt['dataroot_lq'], readonly=True, lock=False, readahead=False,
                                meminit=False)

    def expand2square(timg,factor=16.0):
        h, w, _ = timg.shape

        X = int(math.ceil(max(h,w)/float(factor))*factor)

        img_expand = np.zeros((X,X,3))
        mask = np.ones((X,X,1))

        # print(mask.shape)
        # print((X - h)//2, (X - h)//2+h, (X - w)//2, (X - w)//2+w)

        img_expand[((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w),:] = timg
        mask = mask[((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w),:]
        
        return img_expand

    def __getitem__(self, index):
        if self.data_type == 'lmdb' and (self.gt_env is None or self.lq_env is None):
            self._init_lmdb()
        gt_path, lq_path = None, None
        scale = self.opt['scale']

        # get LQ image(haze image)
        lq_path = self.paths_lq[index]
        resolution = [int(s) for s in self.sizes_lq[index].split('_')
                        ] if self.data_type == 'lmdb' else None
        img_lq = util.read_img(self.lq_env, lq_path, resolution)
        # change color space if necessary
        if self.opt['color']:  # change color space if necessary
            img_lq = util.channel_convert(img_lq.shape[2], self.opt['color'],
                                          [img_lq])[0]  # TODO during val no definition

        # get GT image(haze-free)
        gt_name = lq_path.split('/')[-1].split('_')[0]
        # GT_path = self.paths_GT[index]
        gt_path = os.path.join(self.opt['dataroot_gt'], '{}.png'.format(gt_name))
        resolution = [int(s) for s in self.sizes_lq[index].split('_')
                      ] if self.data_type == 'lmdb' else None
        img_gt = util.read_img(self.gt_env, gt_path, resolution)

        H_gt, W_gt, _ = img_gt.shape
        H_lq, W_lq, _ = img_lq.shape

        if H_gt != H_lq or W_gt != W_lq:
            crop_size_H = np.abs(H_lq-H_gt)//2
            crop_size_W = np.abs(W_lq-W_gt)//2
            img_gt = img_gt[crop_size_H:-crop_size_H, crop_size_W:-crop_size_W, :]

        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size'] if self.opt['gt_size'] is not None else self.opt['hq_size']
            H, W, _ = img_gt.shape
            H, W, C = img_lq.shape
            lq_size = gt_size // scale

            # randomly crop
            rnd_h = random.randint(0, max(0, H - lq_size))
            rnd_w = random.randint(0, max(0, W - lq_size))
            img_lq = img_lq[rnd_h:rnd_h + lq_size, rnd_w:rnd_w + lq_size, :]
            rnd_h_gt, rnd_w_gt = int(rnd_h * scale), int(rnd_w * scale)
            img_gt = img_gt[rnd_h_gt:rnd_h_gt + gt_size, rnd_w_gt:rnd_w_gt + gt_size, :]

            # augmentation - flip, rotate
            img_lq, img_gt = util.augment([img_lq, img_gt], self.opt['use_flip'],
                                          self.opt['use_rot'])


        if self.opt['color']:  # change color space if necessary
            img_gt = util.channel_convert(img_gt.shape[2], self.opt['color'], [img_gt])[0]
        
        if self.opt['phase'] != 'train':
            # print("img_gt:",img_gt.shape)

            # Uncomment these for validation when train Uformer
            # if self.opt['train_net'] == 'Uformer':
            #     # expand picture to square
            #     factor = 128
            #     H_gt, W_gt, _ = img_gt.shape
            #     X = int(math.ceil(max(H_gt,W_gt)/float(factor))*factor)
            #     # print("X:",X)
            #     img_expandgt = np.zeros((X,X,3))
            #     img_expandlq = np.zeros((X,X,3))
            #     mask = np.ones((X,X,1))
            #     # print(img_expandgt.shape,mask.shape)
            #     # print((X - H_gt)//2, (X - H_gt)//2+H_gt, (X - W_gt)//2, (X - W_gt)//2+W_gt)

            #     img_expandgt[((X - H_gt)//2):((X - H_gt)//2 + H_gt),((X - W_gt)//2):((X - W_gt)//2 + W_gt),:] = img_gt
            #     img_expandlq[((X - H_gt)//2):((X - H_gt)//2 + H_gt),((X - W_gt)//2):((X - W_gt)//2 + W_gt),:] = img_lq
            #     mask = mask[((X - H_gt)//2):((X - H_gt)//2 + H_gt),((X - W_gt)//2):((X - W_gt)//2 + W_gt),:]
            #     img_gt = img_expandgt
            #     img_lq = img_expandlq
                
            if img_lq.ndim == 2:
                img_lq = np.expand_dims(img_lq, axis=2)
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_gt.shape[2] == 3:
            img_gt = img_gt[:, :, [2, 1, 0]]
            img_lq = img_lq[:, :, [2, 1, 0]]
        img_gt = torch.from_numpy(np.ascontiguousarray(np.transpose(img_gt, (2, 0, 1)))).float()
        img_lq = torch.from_numpy(np.ascontiguousarray(np.transpose(img_lq, (2, 0, 1)))).float()

        if lq_path is None:
            lq_path = gt_path
        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths_lq)
        # return len(self.paths_gt)