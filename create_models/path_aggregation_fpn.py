'''
@File: path_aggregation_fpn.py
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

from .feature_pyramid_network import *


def _init_weight(m):
    if type(m)==nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)
class PAFPN(FeaturePyramidNetwork):
    '''
    Path Aggregation Network for Instance Segmentation
    https://arxiv.org/abs/1803.01534
    '''
    def __init__(self,in_channel_list,out_channels,extra_block=None):
        super(PAFPN,self).__init__(in_channel_list,out_channels,extra_block)
        self.downsample_convs = nn.ModuleList()
        self.pafpn_convs = nn.ModuleList()
        for i in range(len(in_channel_list)-1):
            d_conv = nn.Conv2d(out_channels,out_channels,3,stride=2,padding=1)
            pafpn_conv = nn.Conv2d(out_channels,out_channels,3,padding=1)
            self.downsample_convs.append(d_conv)
            self.pafpn_convs.append(pafpn_conv)
        self.apply(_init_weight)

    def forward(self,inputs):
        assert len(inputs)==len(self.in_channel_list)
        #get outputs-->自顶向下
        names = list(inputs.keys())
        x = list(inputs.values())
        results = []
        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        results.append(self.get_result_from_layer_blocks(last_inner, -1))
        for idx in range(len(x)-2,-1,-1):
            inner_lateral = self.get_result_from_inner_blocks(x[idx],idx)
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner,size=feat_shape,mode="nearest")
            last_inner  = inner_top_down+inner_lateral
            results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))
        #results[从大特征图到小特征图]-->自底向上
        for i in range(len(results)-1):
            results[i+1] = results[i+1]+self.downsample_convs[i](results[i])
        outs = []
        outs.append(results[0])
        outs.extend([self.pafpn_convs[i-1](results[i])
            for i in range(1,len(results))])
        #添加extrablock
        if self.extra_blocks is not None:
            outs,names = self.extra_blocks(outs,names)
        outs = OrderedDict([(k,v) for k,v in zip(names,outs)])
        return outs
class BackboneWithPAFPN(BackboneWithFPN):
    def __init__(self,backbone,return_layers,in_channel_list,out_channels,extra_blocks,re_getter):
        '''
        backbone+fpn
        :param backbone: model or create
        :param return_layers: dict
        :param in_channel_list: list
        :param out_channels: 255 or 85
        :param extra_blocks: bool
        :param re_getter:  bool
        '''
        super(BackboneWithPAFPN,self).__init__(backbone,return_layers,in_channel_list,out_channels,extra_blocks,re_getter)
        if extra_blocks:
            extb = LastLevelMaxPool()
        else:
            extb = None

        self.fpn = PAFPN(in_channel_list=in_channel_list,
                                         out_channels=out_channels,
                                         extra_block=extb)

    def forward(self,x):#Tensor-->dict
        x = self.body(x)
        x = self.fpn(x)
        return x

#
# if __name__ == '__main__':
#     inputs = {'1':torch.rand(1,256,28,28),
#               '2':torch.rand(1,512,14,14),
#               '3':torch.rand(1,1024,7,7)}
#     net = PAFPN(in_channel_list=[256,512,1024],out_channels=255)
#     outs = net(inputs)
#     print(outs.keys())
#     [print(v.shape) for k,v in outs.items()]
#     print(net)
