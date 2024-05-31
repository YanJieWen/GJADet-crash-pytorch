'''
@File: yolov3_Csp.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 4月 22, 2024
@HomePage: https://github.com/YanJieWen
'''

import ast
import numpy as np
import math
import re
from pathlib import Path
import yaml
import contextlib
import copy
import os

import torch
import torch.nn as nn

from model_utils.parse_config import *
from create_models.dynamic_head import DyHead
from .v8_blocks import *

#YOLOV8各个组件：https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L830
#weights_url = https://docs.ultralytics.com/tasks/detect/
#parser_model()



def make_divisible(x, divisor):
    """Returns nearest x divisible by divisor."""
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


# #create modules
def create_modules(cfg_file, nc=5, type='m', ch=3):
    with open(cfg_file, errors="ignore", encoding="utf-8") as f:
        s = f.read()  # string
    if not s.isprintable():
        s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)
    data = yaml.safe_load(s) or {}
    data['scale'] = type
    data["yaml_file"] = str(cfg_file)
    data['nc'] = nc
    data['ch'] = ch
    d = copy.deepcopy(data)
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    if scales:
        scale = d.get('scale')
        if not scale:
            scale = tuple(scales.keys())[0]
        depth, width, max_channels = scales[scale]  # 0.67，0.75，768
    output_filters = [3]  # 第0个是图像的输入
    module_list = nn.ModuleList()
    routs = []  # 统计哪些层会被后续层适用
    m_defs = d['backbone'] + d['head']
    yolo_index = -1
    for i, (f, n, m, args) in enumerate(m_defs):  # from,num_layers,moudle,args
        m = getattr(torch.nn, m[3:]) if "nn." in m else globals()[m]  # 加载计算模块
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # 更新层数
        if m in {Conv, C2f, SPPF}:
            c1, c2 = output_filters[f], args[0]  # 输入维度和输出维度
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)  # 缩放基带
            # print(m,c1,c2,n)
            args = [c1, c2, *args[1:]]
            if m in {Conv, SPPF}:
                ms = m(*args)
                filters = args[1]
                # routs.append(i)
            elif m is C2f:
                args.insert(2, n)
                ms = m(*args)
                filters = args[1]
                # routs.append(i)
        elif m is Dyattn:
            args.insert(0,output_filters[f])
            ms = m(*args)
        elif m is nn.Upsample:
            ms = m(*args)
            # routs.append(i)
        elif m is Concat:
            if not isinstance(f, list):
                raise TypeError('concat must greater two')
            layers = f
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            routs.append([l + i if l < 0 else l for l in layers])
            ms = m()
        elif m is DyHead:
            if not isinstance(f,list):
                raise TypeError('must multiple feats')
            in_channel_list = [output_filters[l+1] for l in f]
            filters = args[0]
            args = [in_channel_list,*args,n]
            routs.append(f)
            ms = m(*args)
        elif m is Detect:
            ms = nn.ModuleList()
            for layer_index in f:
                yolo_index += 1
                strides = [8,16,32]  # 特征图从大到小，anchors从小到大
                anchors = np.array([[10, 13], [16, 30], [33, 23],
                                    [30, 61], [62, 45], [59, 119],
                                    [116, 90], [156, 198], [373, 326]])
                stride = strides[yolo_index]
                nc = args[0]
                nanchor = anchors[yolo_index * 3:(yolo_index + 1) * 3]
                ch = output_filters[layer_index + 1]
                ms.append(m(nanchor, nc, stride, ch))
            routs.append(f)

        module_list.append(ms)
        output_filters.append(filters)  # 需要用到module_list和和routs
    return nn.Sequential(*module_list), routs

#create detectors
def get_yolo_layer(self):
    return [yolo_layer for i, m in enumerate(self.model) if m.__class__.__name__ == 'ModuleList' for yolo_layer in m]


class CSPDarknet(nn.Module):
    def __init__(self, cfg, img_size=(640, 640), verbose=True, pretrained_root='./pretrained/') -> None:
        super().__init__()
        self.input_size = [img_size] * 2 if isinstance(img_size, int) else img_size
        self.model, self.routs = create_modules(cfg)

        _routs = [j for i in self.routs for j in i]
        self.routs_binary = [False] * len(self.model)
        for i in _routs:
            self.routs_binary[i] = True
        # print(self.routs,self.routs_binary)

        eff_info(self) if verbose else None
        self.info(verbose=verbose)
        self.yolo_layers = get_yolo_layer(self)
        if pretrained_root is not None:
            weights_path = os.path.join(pretrained_root,'u-v8m.pt')
            ckpt = torch.load(weights_path, map_location='cpu')
            ckpt = ckpt['model'] if 'model' in ckpt.keys() else ckpt
            pw = {}
            for k,v in ckpt.items():
                if k in self.state_dict().keys():
                    if self.state_dict()[k].numel()==v.numel():
                        pw[k] = v
                    # else:
                    #     raise ValueError(f'{k}:{self.state_dict()[k].shape}-->{v.shape}')
                else:
                    pass
            # for k, v in ckpt.items():
            #     if '22' not in k:
            #         if self.state_dict()[k].numel() == v.numel():
            #             pw[k] = v
            #         else:
            #             raise ValueError(f'{k}:{self.state_dict()[k].shape}-->{v.shape}')
            #     else:
            #         pass
            miss_key, except_keys = self.load_state_dict(pw, strict=False)
            if len(miss_key) != 0 or len(except_keys) != 0:
                print(f'missing_keys:{miss_key}')
                print(f'unexpected_keys:{except_keys}')
        else:
            pass

    def forward(self, x):
        cat_index = 0
        yolo_out, out = [], []
        for i, module in enumerate(self.model):
            name = module.__class__.__name__
            if name == 'Concat':
                rout = self.routs[cat_index]
                x = module([out[index] for index in rout])
                cat_index += 1
            elif name=='DyHead':
                rout = self.routs[cat_index]
                x = module([out[index] for index in rout])
                cat_index += 1
            elif name == 'ModuleList':
                rout = self.routs[cat_index]
                if len(list(set(rout)))==1:#如果特征图以列表形式存储
                    inputs = out[rout[0]]
                else:
                    inputs = [out[index] for index in rout]
                assert isinstance(inputs,list), f'the inputs type is {type(inputs)}'
                for i, input in enumerate(inputs):
                    yolo_out.append(module[i](input))
            else:
                x = module(x)
            out.append(x if self.routs_binary[i] else [])
        yolo_out.reverse()
        if self.training:
            return yolo_out
        else:
            x, p = zip(*yolo_out)
            x = torch.cat(x, 1)
            return x, p

    def info(self, verbose):
        return model_info(self, verbose=verbose)


# if __name__ == '__main__':
#     path = '../cfg/yolov8.yaml'
#     model = CSPDarknet(path)
#     model.train()
#     yolo_out = model(torch.rand(1, 3, 640, 640))
#     [print(x.shape) for x in yolo_out]