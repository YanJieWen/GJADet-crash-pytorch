'''
@File: feature_pyramid_network.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 3月 17, 2024
@HomePage: https://github.com/YanJieWen
'''


from collections import OrderedDict

import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F

from typing import Dict,Tuple,List


class IntermediateLayerGetter(nn.ModuleList):
    __annotations__ = {
        "return_layers": Dict[str, str],
    }
    def __init__(self,model,return_layers):
        if not set(return_layers).issubset([name for name,_ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name,moudle in model.named_children():
            layers[name] = moudle
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break
        super().__init__()
        self.return_layers = orig_return_layers

    def forward(self,x):
        out = OrderedDict()
        for name,m in self.items():
            x = m(x)
            if name in self.return_layers:
                out_name  = self.return_layers[name]
                out[out_name] = x
        return out


class FeaturePyramidNetwork(nn.Module):
    def __init__(self,in_channel_list,out_channels,extra_block=None):
        super(FeaturePyramidNetwork,self).__init__()
        self.in_channel_list = in_channel_list
        self.out_channels = out_channels
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        for in_channel in in_channel_list:
            if in_channel==0:
                continue
            inner_block_module = nn.Conv2d(in_channel,out_channels,1)
            layer_block_module = nn.Conv2d(out_channels,out_channels,3,padding=1)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)
        for m in self.children():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,a=1)
                nn.init.constant_(m.bias,0)
        self.extra_blocks = extra_block


    def get_result_from_inner_blocks(self,x,idx):
        '''
        内部卷积对齐通道维度
        :param x:
        :param idx:
        :return:
        '''
        num_blocks = len(self.inner_blocks)
        if idx<0:
            idx+=num_blocks
        i = 0
        out = x
        for module in self.inner_blocks:
            if i==idx:
                out =module(x)
            i+=1
        return out

    def get_result_from_layer_blocks(self,x,idx):
        '''
        对外部张量进行transformation
        :param x:
        :param idx:
        :return:
        '''
        num_blocks = len(self.layer_blocks)
        if idx<0:
            idx+=num_blocks
        i = 0
        out = x
        for module in self.layer_blocks:
            if i==idx:
                out = module(x)
            i+=1
        return out



    def forward(self,x):#dict
        names = list(x.keys())
        x = list(x.values())
        results = []
        last_inner = self.get_result_from_inner_blocks(x[-1],-1)#内部卷积对齐通道
        results.append(self.get_result_from_layer_blocks(last_inner,-1))
        for idx in range(len(x)-2,-1,-1):#自顶向下
            inner_lateral = self.get_result_from_inner_blocks(x[idx],idx)
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner,size=feat_shape,mode="nearest")
            last_inner  = inner_top_down+inner_lateral
            results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))
        if self.extra_blocks is not None:
            results,names = self.extra_blocks(results,names)
        out = OrderedDict([(k,v) for k,v in zip(names,results)])

        return out



class LastLevelMaxPool(nn.Module):
    def forward(self,x,names):
        names.append('pool')
        x.append(F.max_pool2d(x[-1],1,2,0))
        return x,names

class BackboneWithFPN(nn.Module):
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
        super(BackboneWithFPN,self).__init__()
        if extra_blocks:
            extb = LastLevelMaxPool()
        else:
            extb = None
        if re_getter is True:
            assert return_layers is not None, f'return layers is not defined'
            self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        else:
            self.body = backbone

        self.fpn = FeaturePyramidNetwork(in_channel_list=in_channel_list,
                                         out_channels=out_channels,
                                         extra_block=extb)
        self.out_channels = out_channels

    def forward(self,x):#Tensor-->dict
        x = self.body(x)
        x = self.fpn(x)
        return x