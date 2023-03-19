import numpy as np
import random
import torch
from collections import OrderedDict
from torch.nn import functional as F
from torchvision import utils as vutils

from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.losses.loss_util import get_refined_artifact_map
from basicsr.models.srgan_model import SRGANModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


@MODEL_REGISTRY.register(suffix='basicsr')
class PretrainRdmgridRealESRGANModel(SRGANModel):
    """RealESRGAN Model for Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt):
        super(PretrainRdmgridRealESRGANModel, self).__init__(opt)
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
            self.gt_usm = self.usm_sharpener(self.gt)

            self.kernel1 = data['kernel1'].to(self.device)
            self.kernel2 = data['kernel2'].to(self.device)
            self.sinc_kernel = data['sinc_kernel'].to(self.device)
            # self.repeat_number = self.opt['repeat_number'].to(self.device)
            self.grid_size = self.opt['grid_size']


            c, ori_h, ori_w = self.gt.size()[1:4]

            # ----------------------- The first degradation process ----------------------- #
            list = []
            for i in range(0,6):
                # blur
                out = filter2D(self.gt_usm, self.kernel1)
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
                    out = filter2D(out, self.sinc_kernel)

                # clamp and round
                degra_img = torch.clamp((out * 255.0).round(), 0, 255) / 255.

                # when batchsize >2, uncomment these
                # degra_img = torch.squeeze(degra_img, 0)

                list.append(degra_img)
            degra_imgStack = torch.stack(list,dim=0)
            # print("degra_imgStack.size:",degra_imgStack.size())
            grid_size = random.choice(self.grid_size)
            degra_grids, num_grids = self.grid_partition(degra_imgStack,grid_size)
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
            degra_shffle_imgStack = self.grid_reverse(degra_grids, grid_size, ori_h // self.opt['scale'], 
                ori_w // self.opt['scale']).permute(1, 0, 2, 3, 4).reshape(-1, c, ori_h // self.opt['scale'], ori_w // self.opt['scale'])
            # print("degra_shffle_imgStack:",degra_shffle_imgStack.shape)

            self.lq = degra_shffle_imgStack.to(self.device)

            # repeat gt for eight times
            lt=[]
            for i in range(0,self.gt.size(0)):
                gt_aug = self.gt[i].unsqueeze_(0).repeat(6,1,1,1)
                gt_aug = gt_aug.unsqueeze_(0)
                for t in gt_aug:
                    lt.append(t)

            self.gt = torch.cat(lt,dim=0).to(self.device)
            
            # # Visualize the gt data
            # for i in range(len(self.gt)):
            #     print("aug_gt")
            #     # print(self.gt[i].shape)
            #     # (3,128,128)
            #     # 复制一份
            #     input_tensor = self.gt[i].clone().detach()
            #     # 到cpu
            #     input_tensor = input_tensor.to(torch.device('cpu'))
            #     vutils.save_image(input_tensor, '/home/yqliu/projects/ClassSwin/BasicSR/results/test_pic/gt_{}.png'.format(i))

            # # Visualize the lq data
            # for i in range(len(self.lq)):
            #     print("aug_lq")
            #     input_tensor = self.lq[i].clone().detach()
            #     # 到cpu
            #     input_tensor = input_tensor.to(torch.device('cpu'))
            #     vutils.save_image(input_tensor, '/home/yqliu/projects/ClassSwin/BasicSR/results/test_pic/lq_{}.png'.format(i))


            # random crop
            gt_size = self.opt['gt_size']
            (self.gt, self.gt_usm), self.lq = paired_random_crop([self.gt, self.gt_usm], self.lq, gt_size,
                                                                 self.opt['scale'])

            # training pair pool
            self._dequeue_and_enqueue()
            # sharpen self.gt again, as we have changed the self.gt with self._dequeue_and_enqueue
            self.gt_usm = self.usm_sharpener(self.gt)
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
        super(PretrainRdmgridRealESRGANModel, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True

    def optimize_parameters(self, current_iter):
        # usm sharpening
        l1_gt = self.gt_usm
        percep_gt = self.gt_usm
        gan_gt = self.gt_usm
        if self.opt['l1_gt_usm'] is False:
            l1_gt = self.gt
        if self.opt['percep_gt_usm'] is False:
            percep_gt = self.gt
        if self.opt['gan_gt_usm'] is False:
            gan_gt = self.gt

        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)
        if self.cri_ldl:
            self.output_ema = self.net_g_ema(self.lq)

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, l1_gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            if self.cri_ldl:
                pixel_weight = get_refined_artifact_map(self.gt, self.output, self.output_ema, 7)
                l_g_ldl = self.cri_ldl(torch.mul(pixel_weight, self.output), torch.mul(pixel_weight, self.gt))
                l_g_total += l_g_ldl
                loss_dict['l_g_ldl'] = l_g_ldl
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, percep_gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            # gan loss
            fake_g_pred = self.net_d(self.output) #output送给判别器得到的结果，fake_g_pred越趋于1（判别为真）越好
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            # gan_loss的参数(input (Tensor), target_is_real, is_disc: Whether the loss for discriminators or not)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            l_g_total.backward() #把net_g的loss反向传播
            self.optimizer_g.step() #把net_g的参数更新

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # real
        real_d_pred = self.net_d(gan_gt)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True) #希望判别器能够知道真的是真的
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        l_d_real.backward()
        # fake
        fake_d_pred = self.net_d(self.output.detach().clone())  # clone for pt1.9
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True) #希望判别器能够知道假的是假的
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        l_d_fake.backward()
        self.optimizer_d.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
        # 指数滑动平均，好处在于1.平滑数据、2.可以存储近似n个时刻的平均值，而不用在内存中保留n个时刻的历史数据，减少了内存消耗。
        # 但是ema不参与实际的训练过程，是用在测试过程的，作用是使得模型在测试数据上更加健壮，有更好的鲁棒性。
        #   或者是最后save模型时存储ema的值，取最近n次的近似平均值，使模型具备更好的测试指标(accuracy)等，更强的泛化能力。
        # 设 decay=0.999，一个更直观的理解，在最后的 1000 次训练过程中，模型早已经训练完成，
        #   正处于抖动阶段，而滑动平均相当于将最后的 1000 次抖动进行了平均，这样得到的权重会更加 robust。
        self.log_dict = self.reduce_loss_dict(loss_dict)