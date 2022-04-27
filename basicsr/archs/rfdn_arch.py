import functools
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from basicsr.utils.registry import ARCH_REGISTRY


class DepthWiseConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DepthWiseConv, self).__init__()
        # 也相当于分组为1的分组卷积
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class BSConvU(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, padding_mode="zeros", with_norm=False, bn_kwargs=None):
        super().__init__()

        # check arguments
        if bn_kwargs is None:
            bn_kwargs = {}

        # pointwise
        self.add_module("pw", torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
        ))

        # batchnorm
        if with_norm:
            self.add_module("ln", torch.nn.LayerNorm(out_channels, **bn_kwargs))

        # depthwise
        self.add_module("dw", torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=out_channels,
                bias=bias,
                padding_mode=padding_mode,
        ))


class BSConvS(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True,
                 padding_mode="zeros", p=0.25, min_mid_channels=4, with_bn=False, bn_kwargs=None):
        super().__init__()

        # check arguments
        assert 0.0 <= p <= 1.0
        mid_channels = min(in_channels, max(min_mid_channels, math.ceil(p * in_channels)))
        if bn_kwargs is None:
            bn_kwargs = {}

        # pointwise 1
        self.add_module("pw1", torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        ))

        # batchnorm
        if with_bn:
            self.add_module("ln1", torch.nn.LayerNorm(out_channels, **bn_kwargs))

        # pointwise 2
        self.add_module("pw2", torch.nn.Conv2d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        ))

        # batchnorm
        if with_bn:
            self.add_module("ln2", torch.nn.LayerNorm(out_channels, **bn_kwargs))

        # depthwise
        self.add_module("dw", torch.nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=out_channels,
            bias=bias,
            padding_mode=padding_mode,
        ))

    def _reg_loss(self):
        W = self[0].weight[:, :, 0, 0]
        WWt = torch.mm(W, torch.transpose(W, 0, 1))
        I = torch.eye(WWt.shape[0], device=WWt.device)
        return torch.norm(WWt - I, p="fro")


class ESA(nn.Module):
    def __init__(self, num_feat=50, conv=nn.Conv2d, p=0.25):
        super(ESA, self).__init__()
        f = num_feat // 4
        BSConvS_kwargs = {}
        if conv.__name__ == 'BSConvS':
            BSConvS_kwargs = {'p': p}
        self.conv1 = nn.Conv2d(num_feat, f, 1, 1, 0)
        self.conv_f = nn.Conv2d(f, f, 1, 1, 0)
        self.maxPooling = nn.MaxPool2d(kernel_size=7, stride=3)
        self.conv_max = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv2 = nn.Conv2d(f, f, 3, 2, 0)
        self.conv3 = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv3_ = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv4 = nn.Conv2d(f, num_feat, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()
        self.GELU = nn.GELU()

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = self.maxPooling(c1)
        v_range = self.GELU(self.conv_max(v_max))
        c3 = self.GELU(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m


class RFDB(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25, conv=nn.Conv2d, p=0.25):
        super(RFDB, self).__init__()
        BSConvS_kwargs = {}
        if conv.__name__ == 'BSConvS':
            BSConvS_kwargs = {'p': p}

        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels

        self.c1_d = nn.Conv2d(in_channels, self.dc, 1, 1, 0)
        self.c1_r = conv(in_channels, self.rc, kernel_size=3,  **BSConvS_kwargs)
        self.c2_d = nn.Conv2d(self.remaining_channels, self.dc, 1, 1, 0)
        self.c2_r = conv(self.remaining_channels, self.rc, kernel_size=3, **BSConvS_kwargs)
        self.c3_d = nn.Conv2d(self.remaining_channels, self.dc, 1, 1, 0)
        self.c3_r = conv(self.remaining_channels, self.rc, kernel_size=3, **BSConvS_kwargs)

        self.c4 = conv(self.remaining_channels, self.dc, kernel_size=3, **BSConvS_kwargs)
        self.act = nn.GELU()

        self.c5 = nn.Conv2d(self.dc * 4, in_channels, 1, 1, 0)
        self.esa = ESA(in_channels, conv)

    def forward(self, input):

        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1 + input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2 + r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3 + r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out_fused = self.esa(self.c5(out))

        return out_fused


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


@ARCH_REGISTRY.register()
class RFDN(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=50, num_block=4, num_out_ch=3, upscale=4, conv='DepthWiseConv', p=0.25,
                 img_range=255, rgb_mean=(0.4029, 0.4484, 0.4680)):
        super(RFDN, self).__init__()
        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.fea_conv = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        print(conv)
        if conv == 'DepthWiseConv':
            self.conv = DepthWiseConv
        elif conv == 'BSConvU':
            self.conv = BSConvU
        elif conv == 'BSConvS':
            self.conv = BSConvS
        else:
            self.conv = nn.Conv2d
        # RFDB_block_f = functools.partial(RFDB, in_channels=num_feat, conv=self.conv, p=p)
        # RFDB_trunk = make_layer(RFDB_block_f, num_block)
        self.B1 = RFDB(in_channels=num_feat, conv=self.conv, p=p)
        self.B2 = RFDB(in_channels=num_feat, conv=self.conv, p=p)
        self.B3 = RFDB(in_channels=num_feat, conv=self.conv, p=p)
        self.B4 = RFDB(in_channels=num_feat, conv=self.conv, p=p)


        self.c1 = nn.Conv2d(num_feat * num_block, num_feat, 1, 1, 0)
        self.GELU = nn.GELU()

        self.c2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.c3 = nn.Conv2d(num_feat, num_out_ch * (upscale ** 2), 3, 1, 1)
        self.pixel = nn.PixelShuffle(upscale)


    def forward(self, input):
        # input = torch.cat([input, input, input], dim=1)
        self.mean = self.mean.type_as(input)
        input = (input - self.mean) * self.img_range
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)

        out_B = self.c1(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_B = self.GELU(out_B)

        out_lr = self.c2(out_B) + out_fea

        output = self.c3(out_lr)
        output = self.pixel(output)
        output = output / self.img_range + self.mean

        return output

