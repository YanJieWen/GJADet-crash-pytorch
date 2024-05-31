'''
@File: vgg.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 3月 17, 2024
@HomePage: https://github.com/YanJieWen
'''
import torch.nn as nn
import torch

import os
import os.path as osp
import importlib

from torchvision.models.feature_extraction import create_feature_extractor
from .Yolov3_spp import *
from .feature_pyramid_network import BackboneWithFPN

# official pretrain weights
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}



class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=False):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)
        # N x 512*7*7
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_features(cfg: list):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg(model_name="vgg16", **kwargs):
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    cfg = cfgs[model_name]

    model = VGG(make_features(cfg), **kwargs)
    return model


class VGG16_Yolo(nn.Module):
    def __init__(self,model_name='vgg16',fpn_type='BackboneWithFPN',img_size=(416,416),in_channel_list=[256,512,512],nc=80,strides=[8,16,32],
                 anchors_wh=[[10,13],[16,30],[33,23],[30,61],[62,45],[59,119],[116,90],[156,198],[373,326]],verbose=True,pretrained_root='./pretrained'):
        super(VGG16_Yolo,self).__init__()
        backbone = vgg(model_name)
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
        return_layers = {'features.16':'0',#8
                     'features.23':'1',#16
                     'features.30':'2'} #32
        anchors = torch.as_tensor(np.array(anchors_wh).reshape(-1,2))
        backbone = create_feature_extractor(backbone,return_layers)
        self.conved_trans = nn.ModuleList([nn.Identity() for _ in in_channel_list])
        self.yolo_layers = nn.ModuleList([YOLOLayer(anchor,nc,img_size,strides[i]) for i,anchor in enumerate(anchors.chunk(3,dim=0))])
        if fpn_type is not None:
            fpn_files = [osp.splitext(osp.basename(x))[0] for x in os.listdir('./create_models/') if not x.startswith('Yolov3')]
            arch_libs = [importlib.import_module(f'create_models.{x}') for x in fpn_files]
            fpn_cfg = {'backbone':backbone,'return_layers':return_layers,'in_channel_list':in_channel_list,
                       'out_channels':(nc+5)*3,'extra_blocks':False,'re_getter':False}
            self.backbonewithfpn = dynamica_install(arch_libs,fpn_type,fpn_cfg)
            # self.backbonewithfpn = BackboneWithPAFPN(backbone,return_layers,in_channel_list,
            #                                        (nc+5)*3,extra_blocks=False,re_getter=False)
        else:
            self.backbonewithfpn = backbone
            self.conved_trans = nn.ModuleList(
                [nn.Conv2d(in_c, (nc + 5) * 3, kernel_size=1) for in_c in in_channel_list])
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
    # fpn_files = [osp.splitext(osp.basename(x))[0] for x in os.listdir('../create_models/')]
    # arch_libs = [importlib.import_module(f'create_models.{x}') for x in fpn_files]
    # print(arch_libs)
#     net = VGG16_Yolo(if_fpn=True)
#     img = torch.randn(1,3,224,224)
#     net.eval()
#     # net.train()
#     # [print(out.shape) for out in net(img)]
#     print(net(img)[0].shape)
#     [print(out.shape) for out in net(img)[1]]