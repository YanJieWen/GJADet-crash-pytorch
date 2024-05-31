'''
@File: Yolov3_Convnext.py
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

import importlib
import os
import os.path as osp


from torchvision.models.feature_extraction import create_feature_extractor
from .Yolov3_spp import *
from model_utils.parse_config import dynamica_install,model_info,eff_info

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_rate (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]

        x = shortcut + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans: int = 3, num_classes: int = 1000, depths: list = None,
                 dims: list = None, drop_path_rate: float = 0., layer_scale_init_value: float = 1e-6,
                 head_init_scale: float = 1.):
        super().__init__()
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                             LayerNorm(dims[0], eps=1e-6, data_format="channels_first"))
        self.downsample_layers.append(stem)

        # 对应stage2-stage4前的3个downsample
        for i in range(3):
            downsample_layer = nn.Sequential(LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                                             nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2))
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        # 构建每个stage中堆叠的block
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_rate=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value)
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.2)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.head(x)
        return x

cfgs ={'convnext_tiny':{'depths':[3,3,9,3],'dims':[96,192,384,768]},
       'convnext_small':{'depths':[3,3,27,3],'dims':[96, 192, 384, 768]},
       'convnext_base':{'depths':[3,3,27,3],'dims':[128, 256, 512, 1024]}}

model_urls = {'convnext_tiny':'https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth',
              'convnext_small':'https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth',
              'convnext_base':'https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth'}
def convnext(model_name='convnext_base',**kwargs):
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    cfg = cfgs[model_name]
    model = ConvNeXt(**cfg)
    return model

class Convnext_Yolo(nn.Module):
    def __init__(self, model_name='convnext_base', fpn_type='BackboneWithFPN', img_size=(416, 416), in_channel_list=[256, 512, 1024], nc=80,
                 strides=[8, 16, 32],anchors_wh=[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198],
                [373, 326]], verbose=True, pretrained_root='./pretrained'):
        super(Convnext_Yolo, self).__init__()
        backbone = convnext(model_name)
        if pretrained_root is not None:
            weights_path = [os.path.join(pretrained_root, x) for x in os.listdir(pretrained_root) if
                            x.startswith(model_name)]
            if len(weights_path) == 0:
                print(f'warning: Not found pre-trained weights for {model_name}')
            else:
                miss_keys, except_keys = backbone.load_state_dict(torch.load(weights_path[0])['model'], strict=False)
                if len(miss_keys) != 0 or len(except_keys) != 0:
                    print("missing_keys: ", miss_keys)
                    print("unexpected_keys: ", except_keys)

        return_layers = {'stages.1': '0',  # 8
                         'stages.2': '1',  # 16
                         'stages.3': '2'}  # 32
        anchors = torch.as_tensor(np.array(anchors_wh).reshape(-1, 2))
        backbone = create_feature_extractor(backbone, return_layers)
        self.conved_trans = nn.ModuleList([nn.Identity() for _ in in_channel_list])
        self.yolo_layers = nn.ModuleList(
            [YOLOLayer(anchor, nc, img_size, strides[i]) for i, anchor in enumerate(anchors.chunk(3, dim=0))])
        if fpn_type is not None:
            fpn_files = [osp.splitext(osp.basename(x))[0] for x in os.listdir('./create_models/') if not x.startswith('Yolov3')]
            arch_libs = [importlib.import_module(f'create_models.{x}') for x in fpn_files]
            fpn_cfg = {'backbone': backbone, 'return_layers': return_layers, 'in_channel_list': in_channel_list,
                       'out_channels': (nc + 5) * 3, 'extra_blocks': False, 're_getter': False}
            self.backbonewithfpn = dynamica_install(arch_libs, fpn_type, fpn_cfg)
        else:
            self.backbonewithfpn = backbone
            self.conved_trans = nn.ModuleList(
                [nn.Conv2d(in_c, (nc + 5) * 3, kernel_size=1) for in_c in in_channel_list])
        # 统计模型的结构以及效率相关信息
        self.info(verbose=verbose)
        eff_info(self) if verbose else None

    def forward(self, x):
        '''
        YOLOpipline
        :param x: b,c,h,w
        :return: train:[b,na,ny,nx,no]*nf;eval:[b,numa,no]解码后的所有anchors在原图的位置信息，[b,na,ny,nx,no]*nf预测信息
        '''
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


if __name__ == '__main__':
    img = torch.randn(1,3,224,224)
    # net = convnext('convnext_base')
    # return_layers = {'stages.1': '0',
    #                   'stages.2':'1',
    #                   'stages.3':'2'}
    # new_net = create_feature_extractor(net,return_layers)
    # [print(f'{k}-->{v.shape}') for k,v in new_net(img).items()]
    net = Convnext_Yolo()
    net.eval()
    x,p = net(img)
    # print(x.shape)
    [print(x.shape) for x in p]


