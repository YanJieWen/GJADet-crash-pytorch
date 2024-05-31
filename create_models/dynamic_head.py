'''
@File: dynamic_head.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 3月 18, 2024
@HomePage: https://github.com/YanJieWen
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from .feature_pyramid_network import BackboneWithFPN,LastLevelMaxPool

class DyHeadBlock(nn.Module):
    def __init__(self,in_channels,out_channels,modulation=True):
        '''

        :param in_channels: input channels
        :param out_channels: output channels
        :param zero_init_offset: whether use zero init for spatial conv
        '''
        super(DyHeadBlock,self).__init__()
        self.spatial_conv_high = DeformConv2d(in_channels,out_channels,modulation=modulation)
        self.spatial_conv_mid = DeformConv2d(in_channels,out_channels,modulation=modulation)
        self.spatial_conv_low = DeformConv2d(in_channels,out_channels,stride=2,modulation=modulation)

        self.scale_attn_module = nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(out_channels,1,1),
                                               nn.ReLU(inplace=True),Hsigmoid(inplace=True))
        self.task_attn_module = DyRelu(out_channels)

        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self,x):
        '''
        dyheadblock
        :param x: List[Tensor]每一层的特征排列从大到小
        :return: List[Tensor]
        '''
        outs = []
        for level in range(len(x)):
            mid_feat = self.spatial_conv_mid(x[level])
            sum_feat = mid_feat*(self.scale_attn_module(mid_feat))
            summed_levels = 1
            if level>0:
                low_feat = self.spatial_conv_low(x[level-1])
                sum_feat += low_feat*self.scale_attn_module(low_feat)
                summed_levels += 1
            if level<len(x)-1:
                high_feat = F.interpolate(self.spatial_conv_high(x[level+1]),
                                          size=x[level].shape[-2:],
                                          mode='bilinear',
                                          align_corners=True)
                sum_feat += high_feat*self.scale_attn_module(high_feat)
                summed_levels += 1
            outs.append(self.task_attn_module(sum_feat/summed_levels))
        return outs


class DyHead(nn.Module):
    '''
    Dynamic Head: Unifying Object Detection Heads with Attentions
    '''
    def __init__(self,in_channel_list,out_channels,num_blocks=6,extra_block=None):
        super(DyHead,self).__init__()
        self.in_channel_list = in_channel_list
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.align_dim_layers = nn.ModuleList([nn.Conv2d(in_channel_list[i],out_channels,1,1)for i in range(len(in_channel_list))])

        dyhead_blocks = []
        for i in range(num_blocks):
            # in_channels = aligen_dims if i==0 else self.out_channels
            in_channels = self.out_channels
            dyhead_blocks.append(DyHeadBlock(in_channels=in_channels,out_channels=self.out_channels))
        self.dyhead_blocks = nn.Sequential(*dyhead_blocks)

        for m in self.children():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,a=1)
                nn.init.constant_(m.bias,0)
        self.extra_blocks = extra_block

    def forward(self,inputs):
        assert isinstance(inputs,(tuple,list,dict))
        if isinstance(inputs,(tuple,list)):
            x = inputs
        elif isinstance(inputs,dict):
            names = list(inputs.keys())
            x = list(inputs.values())
        else:
            pass
        x = [self.align_dim_layers[i](x[i]) for i in range(len(x))]
        assert isinstance(inputs,(tuple,list,dict)),'The inputs must be multiple!'
        results = self.dyhead_blocks(x)
        if self.extra_blocks is not None:
            results,names = self.extra_blocks(results,names)
        if isinstance(inputs,dict):
            out = OrderedDict([(k,v) for k,v in zip(names,results)])
        else:
            out = results
        return out

class BackboneWithDyhead(BackboneWithFPN):
    def __init__(self,backbone,return_layers,in_channel_list,out_channels,extra_blocks,re_getter):
        super(BackboneWithDyhead,self).__init__(backbone,return_layers,in_channel_list,out_channels,extra_blocks,re_getter)
        if extra_blocks:
            extb = LastLevelMaxPool()
        else:
            extb = None
        self.fpn = DyHead(in_channel_list=in_channel_list,
                             out_channels=out_channels,
                             extra_block=extb)

    def forward(self,x):
        x = self.body(x)
        x = self.fpn(x)
        return x





class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)#offset
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)#mask
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)#[b,h,w,2n](x,y)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset,ks):
        '''

        :param x_offset: b,c,h,w,n
        :param ks: kenel size
        :return: b,c,hk,wk
        '''
        b,c,h,w,n = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, n, ks)],dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)
        return x_offset

class DyRelu(nn.Module):
    '''
    https://arxiv.org/abs/2003.10027: Dynamic ReLU
    offical version: https://github.com/microsoft/DynamicHead/blob/master/dyhead/dyrelu.py
    '''
    def __init__(self,channels,ratio=4):
        super(DyRelu,self).__init__()
        self.channels = channels
        self.expansion = 4
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=channels,out_channels=int(channels/ratio),kernel_size=1,stride=1,bias=True),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=int(channels/ratio),out_channels=channels*self.expansion,kernel_size=1,stride=1),
                                   Hsigmoid(inplace=True))
        self.hsigmoid = Hsigmoid(inplace=True)

    def forward(self,x):
        coeffs = self.global_avgpool(x)
        coeffs = self.conv1(coeffs)
        coeffs = self.conv2(coeffs)-0.5#value range[-0.5,0.5]
        a1,b1,a2,b2 = torch.split(coeffs,self.channels,dim=1)
        a1 = a1*2.+1. #value range[-1,1]
        a2 = a2*2.#value range[-1,1]
        out = torch.max(x*a1+b1,x*a2+b2)
        return out



class Hsigmoid(nn.Module):
    def __init__(self, inplace=True, h_max=1):
        super(Hsigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
        self.h_max = h_max

    def forward(self, x):
        return self.relu(x + 3) * self.h_max / 6#bias=3,divisor=6

# if __name__ == '__main__':
#     mdcn = DyHead(in_channel_list=[16,32,64],out_channels=256)
#     feats = [torch.rand(1,16,48,48),torch.rand(1,32,24,24),torch.rand(1,64,12,12)]
#     [print(feat.shape) for feat in mdcn(feats)]