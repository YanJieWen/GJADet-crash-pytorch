'''
@File: Yolov3_resnet.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 3月 18, 2024
@HomePage: https://github.com/YanJieWen
'''


import torch.nn as nn
import torch
import numpy as np


import os
import os.path as osp
import importlib

from torchvision.models.feature_extraction import create_feature_extractor
from .Yolov3_spp import YOLOLayer
from model_utils.parse_config import dynamica_install,model_info,eff_info


model_urls = {
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x

cfgs = {'resnet34' : [BasicBlock,[3, 4, 6, 3],1000,False],
     'resnet50' : [Bottleneck,[3, 4, 6, 3],1000,False],
     'resnet101' : [Bottleneck,[3,4,23,3],1000,False],
     }

def resnet(model_name='resnet34',**kwargs):
    assert model_name in cfgs, 'warning: The model name is not in the Dict'
    cfg = cfgs[model_name]
    model = ResNet(*cfg)
    return model

#[10,13],[16,30],[33,23],[30,61],[62,45],[59,119],[116,90],[156,198],[373,326]
#[2,2],[3,3],[5,3],[5,4],[5,6],[9,10],[10,11],[11,11],[21,21]
class Resnet50_Yolo(nn.Module):
    def __init__(self,model_name='resnet50',fpn_type='BackboneWithFPN',img_size=(416,416),in_channel_list=[512,1024,2048],nc=80,strides=[8,16,32],
                 anchors_wh=[[10,13],[16,30],[33,23],[30,61],[62,45],[59,119],[116,90],[156,198],[373,326]],verbose=True,pretrained_root='./pretrained'):
        super(Resnet50_Yolo,self).__init__()
        backbone = resnet(model_name)
        #加载预训练权重
        if pretrained_root is not None:
            weights_path = [os.path.join(pretrained_root, x) for x in os.listdir(pretrained_root) if x.startswith(model_name.lower())]
            if len(weights_path)==0:
                print(f'warning: Not found pre-trained weights for {model_name}')
            else:
                miss_keys,except_keys = backbone.load_state_dict(torch.load(weights_path[0]),strict=False)
                if len(miss_keys) != 0 or len(except_keys) != 0:
                    print("missing_keys: ", miss_keys)
                    print("unexpected_keys: ", except_keys)
        return_layers = {'layer2': '0',#8
                         'layer3': '1',#16
                         'layer4': '2'} #32
        anchors = torch.as_tensor(np.array(anchors_wh).reshape(-1,2))
        backbone = create_feature_extractor(backbone,return_layers)
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

            self.conved_trans = nn.ModuleList(
                [nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=3, stride=1, padding=1),
                               nn.LeakyReLU(inplace=True),
                               nn.Conv2d(in_channels=in_c, out_channels=(nc + 5) * 3, kernel_size=1, stride=1)
                                        ) for in_c in in_channel_list])#分类定位器
        #统计模型的结构以及效率相关信息
        self.info(verbose=verbose)
        eff_info(self) if verbose else None

    def forward(self,x):
        '''
        YOLOpipline
        :param x: b,c,h,w
        :return: train:[b,na,ny,nx,no]*nf;eval:[b,numa,no]解码后的所有anchors在原图的位置信息，[b,na,ny,nx,no]*nf预测信息
        '''
        x = self.backbonewithfpn(x)
        yolo_out = []
        for i,yolo_layer in enumerate(self.yolo_layers):
            yolo_out.append(yolo_layer(self.conved_trans[i](x[str(i)])))
        #翻转列表保证和Darknet的输出张量为一致的特征图从小到大
        yolo_out.reverse()
        if self.training:
            return yolo_out
        else:
            x,p = zip(*yolo_out)
            x = torch.cat(x,1)
            return x,p

    def info(self,verbose):
        return model_info(self,verbose=verbose)




# if __name__ == '__main__':
#     # net = resnet('resnet50')
#     # return_layers = {'layer2':'0',
#     #                  'layer3':'1',
#     #                  'layer4':'2'}
#     # new_backbone = create_feature_extractor(net, return_layers)
#     img = torch.randn(1, 3, 224, 224)
#     # [print(f'{int(k)}-->{v.shape}') for k, v in new_backbone(img).items()]
#     net = Resnet50_Yolo(pretrained_root=None,fpn_type=None)
#     net.eval()
#     # net.train()
#     # [print(out.shape) for out in net(img)]
#     # print(net(img)[0].shape)
#     [print(out.shape) for out in net(img)[1]]

