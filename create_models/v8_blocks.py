'''
@File: v8_blocks.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 4月 24, 2024
@HomePage: https://github.com/YanJieWen
'''

import re
from pathlib import Path
import yaml
import contextlib
import copy
import os

import torch
import torch.nn as nn


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):  # Convbase块
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True) -> None:
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class C2f(nn.Module):  # csp模块k=3
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5) -> None:
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # [b,c2,h,w]
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class Bottleneck(nn.Module):  # 标准的残差瓶颈层
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5) -> None:
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5) -> None:
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, dim=1))

class Concat(nn.Module):
    def __init__(self, dimension=1) -> None:
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, dim=self.d)

class Detect(nn.Module):  # 只能针对每个特征图进行检测
    def __init__(self, anchors, nc, stride, ch: int) -> None:
        super().__init__()
        self.anchors = torch.as_tensor(anchors)
        self.stride = stride
        self.na = len(anchors)
        self.ch = ch  # number_of _channels
        self.nc = nc
        self.no = nc + 5
        self.nx, self.ny, self.ng = 0, 0, (0, 0)
        self.anchor_vec = self.anchors / stride  # 基于特征图的尺寸
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)
        self.grid = None
        c2, c3 = max((16, 192 // 4, self.na * 4)), max(192, min(self.nc + 1, 100))
        self.loc_cv = nn.Sequential(Conv(ch, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, self.na * 4, 1))
        self.cls_cv = nn.Sequential(Conv(ch, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, (self.nc + 1) * self.na, 1))

    def create_grid(self, ng=(13, 13), device='cpu'):
        self.nx, self.ny = ng
        self.ng = torch.as_tensor(ng, dtype=torch.float)
        if not self.training:
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device),
                                     torch.arange(self.nx, device=device)])
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()  # [1,1,ny,nx,2]坐标系
        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, x):
        p = torch.cat((self.loc_cv(x), self.cls_cv(x)), 1)  # [b,na*(4+1+nc),h,w]
        bs, _, ny, nx = x.shape
        if (self.nx, self.ny) != (nx, ny) or self.grid is None:
            self.create_grid((nx, ny), x.device)
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # [b,3,h,w,no]
        if self.training:
            return p
        else:
            io = p.clone()
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh
            io[..., :4] *= self.stride
            torch.sigmoid_(io[..., 4:])
            return io.view(bs, -1, self.no), p

class DFL(nn.Module):
    '''
Integral module of Distribution Focal Loss (DFL). 积分头
 Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    '''
    def __init__(self,c1=16):
        super().__init__()
        self.conv =nn.Conv2d(c1,1,1,bias=False).requires_grad_(False)
        x = torch.arange(c1,dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1,c1,1,1))
        self.c1 = c1

    def forward(self,x):
        '''

        :param x: tensor[b,4c1,a]
        :return: [b,4,a]
        '''
        b,_,a = x.shape
        return self.conv(x.view(b,4,self.c1,a).transpose(2,1).softmax(1)).view(b,4,a)

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        assert kernel_size in (3,7),'kernel size must be 3 or 7'
        padding =3 if kernel_size==7 else 1
        self.cv1 = nn.Conv2d(2,1,kernel_size,padding=padding,bias=False)
        self.act = nn.Sigmoid()
    def forward(self,x):
        return x*self.act(self.cv1(torch.cat([torch.mean(x,1,keepdim=True),torch.max(x,1,keepdim=True)[0]],1)))

class ChannelAttention(nn.Module):
    def __init__(self,channels,ratio=4):
        super().__init__()
        self.channels = channels
        self.expansion = 4
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=int(channels / ratio), kernel_size=1, stride=1, bias=True),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=int(channels / ratio), out_channels=channels * self.expansion, kernel_size=1,
                      stride=1),
            Hsigmoid(inplace=True))
        # self.hsigmoid = Hsigmoid(inplace=True)

    def forward(self,x):
        coeffs = self.global_avgpool(x)
        coeffs = self.conv1(coeffs)
        coeffs = self.conv2(coeffs) - 0.5#[-0.5,0.5]
        a1, b1, a2, b2 = torch.split(coeffs, self.channels, dim=1)#[-0.5,0.5]
        a1 = a1 * 2. + 1.#[-1.0, 1.0] + 1.0
        a2 = a2 * 2.# [-1.0, 1.0]
        out = torch.max(x * a1 + b1, x * a2 + b2)
        return out/2
class ScaleAttention(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.scale_attn_module = nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(channels,1,1),
                                               nn.ReLU(inplace=True),Hsigmoid(inplace=True))
    def forward(self,x):
        return x*self.scale_attn_module(x)

class Hsigmoid(nn.Module):
    def __init__(self, inplace=True, h_max=1):
        super(Hsigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
        self.h_max = h_max

    def forward(self, x):
        return self.relu(x + 3) * self.h_max / 6#bias=3,divisor=6

class Dyattn(nn.Module):
    def __init__(self,c1,kernel_size=7):
        super().__init__()
        self.channel_attention =ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)
        self.scale_attention = ScaleAttention(c1)
    def forward(self,x):
        return self.scale_attention(self.spatial_attention(x))
        # return self.channel_attention(self.spatial_attention(x))
        # return self.channel_attention(self.scale_attention(x))
        # return self.channel_attention(self.scale_attention(self.spatial_attention(x)))

