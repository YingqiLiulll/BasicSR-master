import numpy as np
import random
import torch
from torchvision import utils as vutils
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.models.sr_model import SRModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from torch.nn import functional as F

from PIL import Image
import matplotlib.pyplot as plt

@MODEL_REGISTRY.register()
class PretrainRdmgridModel(SRModel):
    """Pretrain Model: Design for more efficient pretraining.

    It mainly performs:
    1. randomly synthesize LQ images to eight types in GPU tensors
    2. 
    """

    def __init__(self, opt):
        super(PretrainRdmgridModel, self).__init__(opt)
        self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts

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
        """Accept data from dataloader, and then augment it 8x times to obtain LQ images.
           Approach: randomly select one type of degradation one time, 
           repeat this option for eight times on every grid.
        """
        if self.is_train:
            # training data synthesis
            self.gt = data['gt'].to(self.device)
            self.kernel = data['kernel'].to(self.device)
            self.degradation_type = self.opt['degradation_type']
            self.grid_size = self.opt['grid_size']
            c, h, w = self.gt.size()[1:4]
            # print("h,w:",h,w)
            # print("self.gt.size:",self.gt.size())

            # ----------------------- The degradation process ----------------------- #
            list = []
            for i in range(0,8):
                degradation = random.choice(self.degradation_type)
                # print("degradation_type:",degradation)

                if degradation == 'blur':
                    degra_img = filter2D(self.gt,self.kernel)
                elif degradation == 'noise':
                    gray_noise_prob = self.opt['gray_noise_prob']
                    if np.random.uniform() < self.opt['gaussian_noise_prob']:
                        degra_img = random_add_gaussian_noise_pt(
                            self.gt, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
                    else:
                        degra_img = random_add_poisson_noise_pt(
                            self.gt,
                            scale_range=self.opt['poisson_scale_range'],
                            gray_prob=gray_noise_prob,
                            clip=True,
                            rounds=False)
                elif degradation == 'jpeg':
                    jpeg_p = self.gt.new_zeros(self.gt.size(0)).uniform_(*self.opt['jpeg_range'])
                    out = torch.clamp(self.gt, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
                    out = self.jpeger(out, quality=jpeg_p)
                    degra_img = torch.clamp((out * 255.0).round(), 0, 255) / 255.

                degra_img = torch.squeeze(degra_img, 0)
                list.append(degra_img)

            degra_imgStack = torch.stack(list,dim=0)
            grid_size = random.choice(self.grid_size)
            print("grid_size:",grid_size)
            # print("degra_imgStack.size:",degra_imgStack.size())
            degra_grids, num_grids = self.grid_partition(degra_imgStack, grid_size)
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
            degra_shffle_imgStack = self.grid_reverse(degra_grids, grid_size, h, w).permute(1, 0, 2, 3, 4).reshape(-1, c, h, w)
            # print("degra_shffle_imgStack:",degra_shffle_imgStack.shape)
            
            self.lq = degra_shffle_imgStack.to(self.device)
            # repeat gt for eight times
            lt=[]
            for i in range(0,self.gt.size(0)):
                gt_aug = self.gt[i].unsqueeze_(0).repeat(8,1,1,1)
                gt_aug = gt_aug.unsqueeze_(0)
                for t in gt_aug:
                    lt.append(t)

            self.gt = torch.cat(lt,dim=0).to(self.device)
            # print("self.gt:",self.gt)

            # Visualize the gt data
            # for i in range(len(self.gt)):
            #     # print(self.gt[i].shape)
            #     # (3,128,128)
            #     # 复制一份
            #     input_tensor = self.gt[i].clone().detach()
            #     # 到cpu
            #     input_tensor = input_tensor.to(torch.device('cpu'))

            #     # 反归一化操作，但这里不需要反归一化
            #     # unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            #     # input_tensor = unorm(input_tensor)

            #     vutils.save_image(input_tensor, '/home/yqliu/projects/ClassSwin/BasicSR/results/test_pic/gt_{}.png'.format(i))

            # Visualize the lq data
            # for i in range(len(self.lq)):
            #     input_tensor = self.lq[i].clone().detach()
            #     # 到cpu
            #     input_tensor = input_tensor.to(torch.device('cpu'))

            #     # 反归一化
            #     # unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            #     # input_tensor = unorm(input_tensor)

            #     vutils.save_image(input_tensor, '/home/yqliu/projects/ClassSwin/BasicSR/results/test_pic/lq_{}.png'.format(i))
            

            # Uncomment these for SR task,

            # self.lq = F.interpolate(self.lq, 
            #     size=(h // self.opt['scale'], w // self.opt['scale']), mode='bicubic').to(self.device)
            # # random crop
            # gt_size = self.opt['gt_size']
            # self.gt, self.lq = paired_random_crop(self.gt, self.lq, gt_size, self.opt['scale'])
            
            # training pair pool
            self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract

        else:
            # for paired training or validation
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)

    def test(self):
        if self.opt['network_g']['type'] == 'SwinIR':
            # pad to multiplication of window_size
            window_size = self.opt['network_g']['window_size']
            scale = self.opt.get('scale', 1)
            mod_pad_h, mod_pad_w = 0, 0
            _, _, h, w = self.lq.size()
            if h % window_size != 0:
                mod_pad_h = window_size - h % window_size
            if w % window_size != 0:
                mod_pad_w = window_size - w % window_size
            img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                with torch.no_grad():
                    self.output = self.net_g_ema(img)
            else:
                self.net_g.eval()
                with torch.no_grad():
                    self.output = self.net_g(img)
                self.net_g.train()

            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]
        else:
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                with torch.no_grad():
                    self.output = self.net_g_ema(self.lq)
            else:
                self.net_g.eval()
                with torch.no_grad():
                    self.output = self.net_g(self.lq)
                self.net_g.train()

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        super(PretrainRdmgridModel, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor