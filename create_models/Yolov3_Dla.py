'''
@File: Yolov3_Dla.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 3月 19, 2024
@HomePage: https://github.com/YanJieWen
'''


import math
from os.path import join

import torch
from torch import nn
import torch.utils.model_zoo as model_zoo

import numpy as np
import importlib
import os
import os.path as osp
from torchvision.models.feature_extraction import create_feature_extractor
from .Yolov3_spp import *

BatchNorm = nn.BatchNorm2d

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = BatchNorm(planes)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = BatchNorm(bottle_planes)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = BatchNorm(bottle_planes)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
        # bottle_planes = dim * cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = BatchNorm(bottle_planes)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation, bias=False,
                               dilation=dilation, groups=cardinality)
        self.bn2 = BatchNorm(bottle_planes)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                BatchNorm(out_channels)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, return_levels=False,
                 pool_size=7, linear_root=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.return_levels = return_levels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            BatchNorm(channels[0]),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,
                           root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)

        self.avgpool = nn.AvgPool2d(pool_size)
        self.fc = nn.Conv2d(channels[-1], num_classes, kernel_size=1,
                            stride=1, padding=0, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                BatchNorm(planes),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                BatchNorm(planes),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        if self.return_levels:
            return y
        else:
            x = self.avgpool(x)
            x = self.fc(x)
            x = x.view(x.size(0), -1)

            return x

cfgs = {'dla34':{'levels':[1, 1, 1, 2, 2, 1],'channels':[16, 32, 64, 128, 256, 512],'block':BasicBlock},
        'dla46_c':{'levels':[1, 1, 1, 2, 2, 1],'channels':[16, 32, 64, 64, 128, 256],'block':Bottleneck},
        'dla46x_c':{'levels':[1, 1, 1, 2, 2, 1],'channels':[16, 32, 64, 64, 128, 256],'block':BottleneckX},
        'dla60x_c':{'levels':[1, 1, 1, 2, 3, 1],'channels':[16, 32, 64, 64, 128, 256],'block':BottleneckX},
        'dla60':{'levels':[1, 1, 1, 2, 3, 1],'channels':[16, 32, 128, 256, 512, 1024],'block':Bottleneck},
        'dla60x':{'levels':[1, 1, 1, 2, 3, 1],'channels':[16, 32, 128, 256, 512, 1024],'block':BottleneckX}}

def dla(model_name='dla60',**kwargs):
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    cfg = cfgs[model_name]
    model = DLA(**cfg)
    return model

class DLA60_Yolo(nn.Module):
    def __init__(self,model_name='dla60',fpn_type='BackboneWithFPN',img_size=(416,416),in_channel_list=[256,512,1024],nc=80,strides=[8,16,32],
                 anchors_wh=[[10,13],[16,30],[33,23],[30,61],[62,45],[59,119],[116,90],[156,198],[373,326]],verbose=True,pretrained_root='./pretrained'):
        super(DLA60_Yolo,self).__init__()
        backbone = dla(model_name)
        # 加载预训练权重
        if pretrained_root is not None:
            weights_path = [os.path.join(pretrained_root, x) for x in os.listdir(pretrained_root) if
                            x.startswith(model_name.lower())]
            if len(weights_path) == 0:
                print(f'warning: Not found pre-trained weights for {model_name}')
            else:
                miss_keys, except_keys = backbone.load_state_dict(torch.load(weights_path[0]), strict=False)
                if len(miss_keys) != 0 or len(except_keys) != 0:
                    print("missing_keys: ", miss_keys)
                    print("unexpected_keys: ", except_keys)
        return_layers = {'level3': '0',
                         'level4': '1',
                         'level5': '2'}
        anchors = torch.as_tensor(np.array(anchors_wh).reshape(-1, 2))
        backbone = create_feature_extractor(backbone, return_layers)
        self.conved_trans = nn.ModuleList([nn.Identity() for _ in in_channel_list])
        self.yolo_layers = nn.ModuleList([YOLOLayer(anchor,nc,img_size,strides[i]) for i,anchor in enumerate(anchors.chunk(3,dim=0))])
        if fpn_type is not None:
            fpn_files = [osp.splitext(osp.basename(x))[0] for x in os.listdir('./create_models/') if
                         not x.startswith('Yolov3')]
            arch_libs = [importlib.import_module(f'create_models.{x}') for x in fpn_files]
            fpn_cfg = {'backbone': backbone, 'return_layers': return_layers, 'in_channel_list': in_channel_list,
                       'out_channels': (nc + 5) * 3, 'extra_blocks': False, 're_getter': False}
            self.backbonewithfpn = dynamica_install(arch_libs, fpn_type, fpn_cfg)
        else:
            self.backbonewithfpn = backbone
            self.conved_trans = nn.ModuleList([nn.Conv2d(in_c, (nc + 5) * 3, kernel_size=1) for in_c in in_channel_list])
        # 统计模型的结构以及效率相关信息
        self.info(verbose=verbose)
        eff_info(self) if verbose else None
    def forward(self,x):
        x = self.backbonewithfpn(x)
        yolo_out = []
        for i, yolo_layer in enumerate(self.yolo_layers):
            yolo_out.append(yolo_layer(self.conved_trans[i](x[str(i)])))
        # 翻转列表保证和Darknet的输出张量为一致的特征图从小到大
        yolo_out.reverse()
        if self.training:
            return yolo_out
        else:
            x, p = zip(*yolo_out)
            x = torch.cat(x, 1)
            return x, p
    def info(self,verbose):
        return model_info(self,verbose=verbose)

# if __name__ == '__main__':
#     net = DLA60_Yolo(fpn_type='BackboneWithDyhead')
#     img = torch.rand(1,3,224,224)
#     x,p  = net(img)
#     net.eval()
#     [print(x.shape) for x in p]
