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

@MODEL_REGISTRY.register()
class PretrainModel(SRModel):
    """Pretrain Model: Design for more efficient pretraining.

    It mainly performs:
    1. randomly synthesize LQ images to eight types in GPU tensors
    2. 
    """

    def __init__(self, opt):
        super(PretrainModel, self).__init__(opt)
        self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts

    # ----------------------- grid partition & reverse ----------------------- #

    def grid_partition(x, grid_size):
        """
        Args:
            x: (B, C, H, W) tensor
            grid_size (int): grid size

        Returns:
            grids: (B, C, num_grids, grid_size, grid_size)
        """
        B, C, H, W = x.size()[0:4]
        x = x.view(B, C, H // grid_size, grid_size, W // grid_size, grid_size)
        # print("x_size:",x.size)
        num_grids = (H // grid_size)*(W // grid_size)
        grids = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, num_grids, grid_size, grid_size)
        # print('grids:',grids.shape)
        return grids, num_grids

    def grid_reverse(grids, grid_size, h, w):
        """
        Args:
            grids: (b, c, num_grids, grid_size, grid_size)
            grid_size (int): grid size
            h (int): Height of image
            w (int): Width of image

        Returns:
            x: (b, c, h, w)
        """
        b,c = grids.size()[0:2]
        x = grids.view(b, c, h // grid_size, w // grid_size, grid_size, grid_size)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(b, c, h, w)

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
            
            h, w = self.gt.size()[2:4]

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
            # print("degra_imgStack.size:",degra_imgStack.size())
            degra_grids, num_grids = self.grid_partition(degra_imgStack,16)
            # return grids: (B, C, num_grids, grid_size, grid_size)

            # shuffle tensor
            # it's more troublesome than directly shuffle on numpy, 
            # you can also change degra_grids from tensor to numpy, and use "np.random.shuffle"
            for i in range(0,num_grids):
                idx = torch.randperm(degra_grids.shape[0])
                current_grid = degra_grids[:,:,i,:,:]
                # print("current_grid:",current_grid.size())
                shuffle_grid = current_grid[idx,:,:,:]
                degra_grids[:,:,i,:,:] = shuffle_grid

            # print("degra_grids_shape:",degra_grids.shape)
            degra_shffle_imgStack = self.grid_reverse(degra_grids, 16, h, w)

            for id in range(0,8):
                self.lq = degra_shffle_imgStack[id].to(self.device)
                # random crop
                gt_size = self.opt['gt_size']
                self.gt, self.lq = paired_random_crop(self.gt, self.lq, gt_size, self.opt['scale'])
                # training pair pool
                self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract
        else:
            # for paired training or validation
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        super(PretrainModel, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True
