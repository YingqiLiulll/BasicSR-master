import numpy as np
import random
import torch
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.models.sr_model import SRModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from torch.nn import functional as F
from torchvision import utils as vutils


@MODEL_REGISTRY.register()
class PretrainRealESRNetModel(SRModel):
    """RealESRNet Model for Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It is trained without GAN losses.
    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt):
        super(PretrainRealESRNetModel, self).__init__(opt)
        self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
        self.usm_sharpener = USMSharp().cuda()  # do usm sharpening
        self.queue_size = opt.get('queue_size', 180)

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    # ----------------------- grid partition & reverse ----------------------- #

    def grid_partition(self, x, grid_size):
        """
        Args:
            x: (N, B, C, H, W) tensor //N:number of degradation augment /B:batchsize
            grid_size (int): grid size

        Returns:
            grids: (N, B, C, num_grids, grid_size, grid_size)
        """
        N, B, C, H, W = x.size()[0:5]
        # print("N, B, C, H, W :", N, B, C, H, W)
        x = x.view(N, B, C, H // grid_size, grid_size, W // grid_size, grid_size)
        # print("x_size:",x.size)
        num_grids = (H // grid_size)*(W // grid_size)
        grids = x.permute(0, 1, 2, 3, 5, 4, 6).contiguous().view(N, B, C, num_grids, grid_size, grid_size)
        # print('grids:',grids.shape)
        return grids, num_grids

    def grid_reverse(self, grids, grid_size, h, w):
        """
        Args:
            grids: (n, b, c, num_grids, grid_size, grid_size)
            grid_size (int): grid size
            h (int): Height of image
            w (int): Width of image

        Returns:
            x: (n, b, c, h, w)
        """
        n,b,c = grids.size()[0:3]
        x = grids.view(n, b, c, h // grid_size, w // grid_size, grid_size, grid_size)
        x = x.permute(0, 1, 2, 3, 5, 4, 6).contiguous().view(n, b, c, h, w)

        return x

    @torch.no_grad()
    def feed_data(self, data):
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images.
        """
        if self.is_train and self.opt.get('high_order_degradation', True):
            # training data synthesis
            self.gt = data['gt'].to(self.device)
            # print("self.gt:",self.gt.shape)
            # USM sharpen the GT images
            if self.opt['gt_usm'] is True:
                self.gt = self.usm_sharpener(self.gt)

            self.kernel1 = data['kernel1'].to(self.device)
            self.kernel2 = data['kernel2'].to(self.device)
            self.sinc_kernel = data['sinc_kernel'].to(self.device)

            c, ori_h, ori_w = self.gt.size()[1:4]


            # ----------------------- The first degradation process ----------------------- #
            list = []
            for i in range(0,9):
                # blur
                out = filter2D(self.gt, self.kernel1)
                # random resize
                updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
                if updown_type == 'up':
                    scale = np.random.uniform(1, self.opt['resize_range'][1])
                elif updown_type == 'down':
                    scale = np.random.uniform(self.opt['resize_range'][0], 1)
                else:
                    scale = 1
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, scale_factor=scale, mode=mode)
                # add noise
                gray_noise_prob = self.opt['gray_noise_prob']
                if np.random.uniform() < self.opt['gaussian_noise_prob']:
                    out = random_add_gaussian_noise_pt(
                        out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
                else:
                    out = random_add_poisson_noise_pt(
                        out,
                        scale_range=self.opt['poisson_scale_range'],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
                out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
                out = self.jpeger(out, quality=jpeg_p)

                # ----------------------- The second degradation process ----------------------- #
                # blur
                if np.random.uniform() < self.opt['second_blur_prob']:
                    out = filter2D(out, self.kernel2)
                # random resize
                updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
                if updown_type == 'up':
                    scale = np.random.uniform(1, self.opt['resize_range2'][1])
                elif updown_type == 'down':
                    scale = np.random.uniform(self.opt['resize_range2'][0], 1)
                else:
                    scale = 1
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(
                    out, size=(int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)), mode=mode)
                # add noise
                gray_noise_prob = self.opt['gray_noise_prob2']
                if np.random.uniform() < self.opt['gaussian_noise_prob2']:
                    out = random_add_gaussian_noise_pt(
                        out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
                else:
                    out = random_add_poisson_noise_pt(
                        out,
                        scale_range=self.opt['poisson_scale_range2'],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False)

                # JPEG compression + the final sinc filter
                # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
                # as one operation.
                # We consider two orders:
                #   1. [resize back + sinc filter] + JPEG compression
                #   2. JPEG compression + [resize back + sinc filter]
                # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
                if np.random.uniform() < 0.5:
                    # resize back + the final sinc filter
                    mode = random.choice(['area', 'bilinear', 'bicubic'])
                    out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                    out = filter2D(out, self.sinc_kernel)
                    # JPEG compression
                    jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                    out = torch.clamp(out, 0, 1)
                    out = self.jpeger(out, quality=jpeg_p)
                else:
                    # JPEG compression
                    jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                    out = torch.clamp(out, 0, 1)
                    out = self.jpeger(out, quality=jpeg_p)
                    # resize back + the final sinc filter
                    mode = random.choice(['area', 'bilinear', 'bicubic'])
                    out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                    # print("out_shape:",out.shape)
                    out = filter2D(out, self.sinc_kernel)

                # clamp and round
                degra_img = torch.clamp((out * 255.0).round(), 0, 255) / 255.
                # print("degra_img:",degra_img.shape)

                # when batchsize >2, uncomment these
                # degra_img = torch.squeeze(degra_img, 0)

                list.append(degra_img)
            degra_imgStack = torch.stack(list,dim=0)
            # print("degra_imgStack.size:",degra_imgStack.size())
            degra_grids, num_grids = self.grid_partition(degra_imgStack,10)
            # return grids: (N, B, C, num_grids, grid_size, grid_size)

            # shuffle tensor
            # it's more troublesome than directly shuffle on numpy, 
            # you can also change degra_grids from tensor to numpy, and use "np.random.shuffle"
            for i in range(0,num_grids):
                idx = torch.randperm(degra_grids.shape[0])
                current_grid = degra_grids[:,:,:,i,:,:]
                # print("current_grid:",current_grid.size())
                shuffle_grid = current_grid[idx,:,:,:,:]
                degra_grids[:,:,:,i,:,:] = shuffle_grid

            # print("degra_grids_shape:",degra_grids.shape)
            degra_shffle_imgStack = self.grid_reverse(degra_grids, 10, ori_h // self.opt['scale'], 
                ori_w // self.opt['scale']).permute(1, 0, 2, 3, 4).reshape(-1, c, ori_h // self.opt['scale'], ori_w // self.opt['scale'])
            # print("degra_shffle_imgStack:",degra_shffle_imgStack.shape)

            self.lq = degra_shffle_imgStack.to(self.device)

            # repeat gt for eight times
            lt=[]
            for i in range(0,self.gt.size(0)):
                gt_aug = self.gt[i].unsqueeze_(0).repeat(9,1,1,1)
                gt_aug = gt_aug.unsqueeze_(0)
                for t in gt_aug:
                    lt.append(t)

            self.gt = torch.cat(lt,dim=0).to(self.device)

            # Visualize the gt data
            # for i in range(len(self.gt)):
            #     # print(self.gt[i].shape)
            #     # (3,128,128)
            #     # 复制一份
            #     input_tensor = self.gt[i].clone().detach()
            #     # 到cpu
            #     input_tensor = input_tensor.to(torch.device('cpu'))
            #     vutils.save_image(input_tensor, '/home/yqliu/projects/ClassSwin/BasicSR/results/test_pic/gt_{}.png'.format(i))

            # Visualize the lq data
            # for i in range(len(self.lq)):
            #     input_tensor = self.lq[i].clone().detach()
            #     # 到cpu
            #     input_tensor = input_tensor.to(torch.device('cpu'))
            #     vutils.save_image(input_tensor, '/home/yqliu/projects/ClassSwin/BasicSR/results/test_pic/lq_{}.png'.format(i))

            # random crop
            gt_size = self.opt['gt_size']
            self.gt, self.lq = paired_random_crop(self.gt, self.lq, gt_size, self.opt['scale'])

            # training pair pool
            self._dequeue_and_enqueue()
            self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract
        else:
            # for paired training or validation
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
                self.gt_usm = self.usm_sharpener(self.gt)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        super(PretrainRealESRNetModel, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True
