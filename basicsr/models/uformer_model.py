import torch
from torch.nn import functional as F

from basicsr.utils.registry import MODEL_REGISTRY
from .sr_model import SRModel
import math


@MODEL_REGISTRY.register()
class UformerModel(SRModel):

    def expand2square(self, timg,factor=16.0):
        _, _, h, w = timg.size()

        X = int(math.ceil(max(h,w)/float(factor))*factor)

        img = torch.zeros(1,3,X,X).type_as(timg) # 3, h,w
        mask = torch.zeros(1,1,X,X).type_as(timg)

        # print(img.size(),mask.size())
        # print((X - h)//2, (X - h)//2+h, (X - w)//2, (X - w)//2+w)
        img[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)] = timg
        mask[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)].fill_(1)
    
        return img, mask, X

    def test(self):
        # pad to square
        # factor=128
        scale = self.opt.get('scale', 1)
        # _, _, h, w = self.lq.size()
        # X = int(math.ceil(max(h,w)/float(factor))*factor)
        # img = torch.zeros(1,3,X,X).type_as(self.lq) # 3, h,w
        # mask = torch.zeros(1,1,X,X).type_as(self.lq)
        # img[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)] = self.lq
        # mask[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)].fill_(1)
        img,mask,X = self.expand2square(self.lq, factor=128)
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
        self.output = self.output[:, :, ((X - h)//2)* scale : ((X - h)//2 + h)* scale, ((X - w)//2):((X - w)//2 + w)* scale]
