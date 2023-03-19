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
class PretrainPairModel(SRModel):
    """Pretrain Pair Model: Baseline, generate pair training data.

    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. 
    """

    def __init__(self, opt):
        super(PretrainPairModel, self).__init__(opt)
        self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts

    @torch.no_grad()
    def feed_data(self, data):
        """Accept data from dataloader, and then add random degradation to obtain LQ images.
           Approach: randomly select one type of degradation one time.
        """
        if self.is_train:
            # training data synthesis
            self.gt = data['gt'].to(self.device)
            self.kernel = data['kernel'].to(self.device)
            self.degradation_type = self.opt['degradation_type']
            
            c, h, w = self.gt.size()[1:4]
            # print("h,w:",h,w)
            # print("self.gt.size:",self.gt.size())

            # ----------------------- The degradation process ----------------------- #
            degradation = random.choice(self.degradation_type)
            # print("degradation_type:",degradation)

            if degradation == 'blur':
                self.lq = filter2D(self.gt,self.kernel)
            elif degradation == 'noise':
                gray_noise_prob = self.opt['gray_noise_prob']
                if np.random.uniform() < self.opt['gaussian_noise_prob']:
                    self.lq = random_add_gaussian_noise_pt(
                        self.gt, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
                else:
                    self.lq = random_add_poisson_noise_pt(
                        self.gt,
                        scale_range=self.opt['poisson_scale_range'],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False)
            elif degradation == 'jpeg':
                jpeg_p = self.gt.new_zeros(self.gt.size(0)).uniform_(*self.opt['jpeg_range'])
                out = torch.clamp(self.gt, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
                out = self.jpeger(out, quality=jpeg_p)
                self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.
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
        super(PretrainPairModel, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True