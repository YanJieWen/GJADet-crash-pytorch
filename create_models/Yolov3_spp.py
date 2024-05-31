'''
@File: Yolov3_spp.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 3月 16, 2024
@HomePage: https://github.com/YanJieWen
'''

import torch
import torch.nn as nn

import numpy as np
import math

from model_utils.parse_config import *


class FeatureConcat(nn.Module):
    def __init__(self,layers):
        super(FeatureConcat,self).__init__()
        self.layers = layers
        self.multiple = len(layers)>1
    def forward(self,x,outputs):
        return torch.cat([outputs[i] for i in self.layers],dim=1) if self.multiple else outputs[self.layers[0]]


class WeightedFeatureFusion(nn.Module):  # weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    """
    将多个特征矩阵的值进行融合(add操作)
    """
    def __init__(self, layers):
        super(WeightedFeatureFusion, self).__init__()
        self.layers = layers  # layer indices
        self.n = len(layers) + 1  # number of layers 融合的特征矩阵个数

    def forward(self, x, outputs):
        # Fusion
        nx = x.shape[1]  # input channels
        for i in range(self.n - 1):
            a = outputs[self.layers[i]]  # feature to add
            na = a.shape[1]  # feature channels
            # Adjust channels
            # 根据相加的两个特征矩阵的channel选择相加方式
            if nx == na:  # same shape 如果channel相同，直接相加
                x = x + a
            elif nx > na:  # slice input 如果channel不同，将channel多的特征矩阵砍掉部分channel保证相加的channel一致
                x[:, :na] = x[:, :na] + a  # or a = nn.ZeroPad2d((0, 0, 0, 0, 0, dc))(a); x = x + a
            else:  # slice feature
                x = x + a[:, :nx]

        return x

def create_modules(m_defs:list, img_size):
    '''
    根据解析的文件定义模块
    :param m_defs:List[Dict]
    :param img_size:
    :return:
    '''
    m_defs = m_defs[1:]
    output_filters = [3]
    module_list = nn.ModuleList()
    routs = []#统计哪些层会被后续使用
    yolo_index = -1
    for i,mdef in enumerate(m_defs):
        ms = nn.Sequential()
        if mdef['type']=='convolutional':
            bn = mdef['batch_normalize']
            filters = mdef['filters']
            k = mdef['size']
            stride = mdef['stride'] if 'stride' in mdef else (mdef['stride_y'],mdef['stride_x'])
            if isinstance(k,int):
                ms.add_module('Conv2d',nn.Conv2d(in_channels=output_filters[-1],
                                                 out_channels=filters,
                                                 kernel_size=k,
                                                 stride=stride,
                                                 padding=k//2 if mdef['pad'] else 0,
                                                 bias=not bn))#有bn的话bias失效
            else:
                raise TypeError('conv2d filter must  be int type')
            if bn:
                ms.add_module('BatchNorm2d',nn.BatchNorm2d(filters))
            else:
                routs.append(i)
            if mdef['activation']=='leaky':
                ms.add_module('activation',nn.LeakyReLU(0.1,inplace=True))
            else:
                pass
        elif mdef['type']=='BatchNorm2d':
            pass

        elif mdef['type']=='maxpool':
            k = mdef['size']
            stride = mdef['stride']
            ms = nn.MaxPool2d(kernel_size=k,stride=stride,padding=(k-1)//2)
        elif mdef['type']=='upsample':
            ms = nn.Upsample(scale_factor=mdef["stride"],mode='nearest')
        elif mdef['type']=='route':
            layers =mdef['layers']
            filters = sum([output_filters[l+1 if l>0 else l ] for l in layers])
            routs.extend([i+l if l<0 else l for l in layers])
            ms = FeatureConcat(layers=layers)
        elif mdef["type"] == "shortcut":
            layers = mdef["from"]
            filters = output_filters[-1]
            routs.append(i + layers[0])
            ms = WeightedFeatureFusion(layers=layers)
        elif mdef["type"] == "yolo":
            yolo_index += 1 #[0,1,2]
            stride = [32, 16, 8]
            ms = YOLOLayer(anchors=mdef["anchors"][mdef["mask"]],  # anchor list,mask用于选取anchors
                                nc=mdef["classes"],  # number of classes
                                img_size=img_size,
                                stride=stride[yolo_index])
            try: #对前一层的conv偏置初始化
                j = -1
                # bias: shape(255,) 索引0对应Sequential中的Conv2d
                # view: shape(3, 85)
                b = module_list[j][0].bias.view(ms.na, -1)
                b.data[:, 4] += -4.5  # obj
                b.data[:, 5:] += math.log(0.6 / (ms.nc - 0.99))  # cls (sigmoid(p) = 1/nc)
                module_list[j][0].bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            except Exception as e:
                print('WARNING: smart bias initialization failure.', e)
        else:
            print("Warning: Unrecognized Layer Type: " + mdef["type"])

        module_list.append(ms)
        output_filters.append(filters)
    route_binary = [False]*len(m_defs)
    for i in routs:
        route_binary[i] = True
    return module_list,route_binary




class YOLOLayer(nn.Module):
    '''
    对YOLOpredictor的输出进行处理
    '''
    def __init__(self,anchors:np.array,nc:int,img_size:tuple,stride:int):
        super(YOLOLayer,self).__init__()
        self.anchors = torch.as_tensor(anchors) if not isinstance(anchors,torch.Tensor) else anchors
        self.stride = stride
        self.na = len(anchors)
        self.nc = nc
        self.no = nc+5
        self.nx,self.ny,self.ng = 0,0,(0,0)
        self.anchor_vec = self.anchors/self.stride
        self.anchor_wh = self.anchor_vec.view(1,self.na,1,1,2)#b,na,h,w,2(wh)
        self.grid = None #[1,1,ny,nx,2]

    def create_grid(self,ng=(13,13),device='cpu'):
        '''
        更新grids信息并生成新的grids参数
        :param ng: 输入特征图的大小（nx,ny）
        :param device:
        :return:
        '''
        self.nx,self.ny = ng
        self.ng = torch.as_tensor(ng,dtype=torch.float)
        if not self.training:
            yv,xv = torch.meshgrid([torch.arange(self.ny,device=device),
                                    torch.arange(self.nx,device=device)])
            self.grid = torch.stack((xv,yv),dim=2).view(1,1,self.ny,self.nx,2).float() #左上角坐标
        if self.anchor_vec.device!=device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)


    def forward(self,p):
        bs,_,ny,nx =p.shape
        if (self.nx,self.ny)!=(nx,ny) or self.grid is None:
            self.create_grid((nx,ny),p.device)
        p = p.view(bs,self.na,self.no,self.ny,self.nx).permute(0,1,3,4,2).contiguous()#[b,na,ny,nx,no]#特征图每个网格有na个anchor，每个anchor no个参数
        if self.training:
            return p
        else:
            io = p.clone()
            io[...,:2] = torch.sigmoid(io[...,:2]) + self.grid
            io[...,2:4] = torch.exp(io[...,2:4]) * self.anchor_wh
            io[...,:4]*=self.stride #换算映射回原来的尺度
            torch.sigmoid_(io[...,4:])#将概率映射到0-1
            return io.view(bs,-1,self.no),p#[b,na*ny*nx,no],#[b,na,ny,nx,no]#多少个调整后的anchors


def get_yolo_layer(self):
    return [i for i,m in enumerate(self.module_list) if m.__class__.__name__=='YOLOLayer']


class Darknet(nn.Module):
    def __init__(self,cfg,img_size=(416,416),verbose=False,pretrained_root='./pretrained'):
        super(Darknet,self).__init__()
        self.input_size = [img_size]*2 if isinstance(img_size,int) else img_size
        self.modules_def = parse_model_config(cfg)
        self.module_list,self.routs = create_modules(self.modules_def,img_size)
        eff_info(self) if verbose else None
        self.yolo_layers = get_yolo_layer(self)
        self.info(verbose=verbose)
        # 加载预训练权重
        if pretrained_root is not None:
            weights_path = os.path.join(pretrained_root, 'yolov3-spp-ultralytics-512.pt')
            ckpt = torch.load(weights_path, map_location='cpu')
            ckpt["model"] = {k: v for k, v in ckpt["model"].items() if self.state_dict()[k].numel() == v.numel()}
            miss_keys, except_keys = self.load_state_dict(ckpt["model"],strict=False)
            # miss_keys, except_keys = self.load_state_dict(torch.load(weights_path)['model'], strict=False)
            if len(miss_keys) != 0 or len(except_keys) != 0:
                print("missing_keys: ", miss_keys)
                print("unexpected_keys: ", except_keys)
    def forward(self,x):
        yolo_out,out = [],[]
        str = ""
        for i,module in enumerate(self.module_list):
            name = module.__class__.__name__
            if name in ["WeightedFeatureFusion", "FeatureConcat"]:
                l = [i-1]+module.layers
                sh = [list(x.shape)] + [list(out[i].shape) for i in module.layers]
                str = ' >> ' + ' + '.join(['layer %g %s' % x for x in zip(l, sh)])
                x = module(x, out)
            elif name=='YOLOLayer':
                yolo_out.append(module(x))
            else:
                x = module(x)

            out.append(x if self.routs[i] else [])
            # print('%g/%g %s -' % (i, len(self.module_list), name), list(x.shape), str)
            # str = ''
        if self.training:
            return yolo_out
        else:#推理模式返回原尺度和非原尺度，并对原尺度的所有修正的anchors信息进行拼接
            x,p = zip(*yolo_out)
            x = torch.cat(x, 1)
            return x, p

    def info(self,verbose):
        return model_info(self,verbose=verbose)


# if __name__ == '__main__':
#     path = '../cfg/yolov3-spp.cfg'
#     # mdefs = parse_model_config(path)
#     # module_list, route_binary = create_modules(mdefs,416)
#     net = Darknet(path,verbose=True)
#     net.train()
#     img = torch.randn(1,3,224,224)
#     [print(out.shape) for out in net(img)]
#     print('1')
