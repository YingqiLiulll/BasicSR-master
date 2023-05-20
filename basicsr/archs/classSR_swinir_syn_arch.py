import functools
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import basicsr.archs.arch_util as arch_util
import torch
from basicsr.archs.synswin_fix0_arch import SynSwinIR_Fix0
from basicsr.archs.swinir_arch import SwinIR
import numpy as np
import time
from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class classSR_3class_swinir_syn(nn.Module):
    def __init__(self, in_chans=3,):
        super(classSR_3class_swinir_syn, self).__init__()
        self.upscale=4
        self.classifier=Classifier()
        self.net1 = SynSwinIR_Fix0(img_size=32, in_chans=in_chans, depths=(6, 6, 6, 6), num_heads=(6, 6, 6, 6),
         embed_dim=60, window_size=8, mlp_ratio=2.,upscale=4, upsampler='pixelshuffledirect', resi_connection='1conv',attn_type='fact_dense_pose3')
        self.net2 = SynSwinIR_Fix0(img_size=32, in_chans=in_chans, depths=(6, 6, 6, 6), num_heads=(6, 6, 6, 6),
         embed_dim=60, window_size=8, mlp_ratio=2.,upscale=4, upsampler='pixelshuffledirect', resi_connection='1conv',attn_type='vanilla_kout')
        self.net3 = SwinIR(img_size=32, in_chans=in_chans, depths=(6, 6, 6, 6), num_heads=(6, 6, 6, 6),
         embed_dim=60, window_size=8, mlp_ratio=2.,upscale=4, upsampler='pixelshuffledirect', resi_connection='1conv')

    def forward(self, x,is_train):
        if is_train:
            for i in range(len(x)):
                # print(x[i].unsqueeze(0).shape)
                # x = [N, C, H, W], len(x)=N
                type = self.classifier(x[i].unsqueeze(0))
                # x[0] = [C, H, W], x[0].unsqueeze(0) = [1, C, H, W]
                # type is probability
                p = F.softmax(type, dim=1)
                p1 = p[0][0]
                p2 = p[0][1]
                p3 = p[0][2]
                # x[0] have three kinds

                out1 = self.net1(x[i].unsqueeze(0))
                out2 = self.net2(x[i].unsqueeze(0))
                out3 = self.net3(x[i].unsqueeze(0))
                # a picture is synthesized from multiple parts

                out = out1 * p1 + out2 * p2 + out3 * p3
                if i == 0:
                    out_res = out
                    type_res = p
                else:
                    out_res = torch.cat((out_res, out), 0)
                    type_res = torch.cat((type_res, p), 0)
                    # concat all x[i] probability
        else:

            for i in range(len(x)):
                type = self.classifier(x[i].unsqueeze(0))

                flag = torch.max(type, 1)[1].data.squeeze()
                p = F.softmax(type, dim=1)
                #flag=np.random.randint(0,2)
                #flag=2
                if flag == 0:
                    out = self.net1(x[i].unsqueeze(0))
                elif flag==1:
                    out = self.net2(x[i].unsqueeze(0))
                elif flag==2:
                    out = self.net3(x[i].unsqueeze(0))
                if i == 0:
                    out_res = out
                    type_res = p
                else:
                    out_res = torch.cat((out_res, out), 0)
                    type_res = torch.cat((type_res, p), 0)

            return out_res, type_res

        return out_res,type_res

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.lastOut = nn.Linear(32, 3)

        # Condtion network
        self.CondNet = nn.Sequential(nn.Conv2d(3, 128, 4, 4), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(128, 32, 1))
        # nn.conv2d(in_channels, out_channels, kernel_size, stride, padding)
        initialize_weights([self.CondNet], 0.1)
    def forward(self, x):
        out = self.CondNet(x)
        out = nn.AvgPool2d(out.size()[2])(out)
        out = out.view(out.size(0), -1) #flatten the last dimension
        out = self.lastOut(out)
        return out

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)