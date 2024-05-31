'''
@File: parse_config.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 3月 16, 2024
@HomePage: https://github.com/YanJieWen
'''


import os
import numpy as np

from thop import profile
from thop import clever_format
import torch


def parse_model_config(path:str):
    '''
    解压YOLOV3的网络
    :param path: str
    :return: [Dict]
    '''
    if not path.endswith('.cfg') or not os.path.exists(path):
        raise FileNotFoundError('the cfg file not exist...')
    with open(path,'r') as r:
        lines = r.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.strip() for x in lines]
    mdefs = []
    for line in lines:
        if line.startswith('['):
            mdefs.append({})
            mdefs[-1]['type'] = line[1:-1].strip()
            if mdefs[-1]['type'] == 'convolutional':
                mdefs[-1]['batch_normalize']=0
        else:
            key,value = line.split('=')
            key = key.strip()
            value = value.strip()

            if key=='anchors':
                value = value.replace(' ','')
                mdefs[-1][key] = np.array([float(x) for x in value.split(',')]).reshape(-1,2)
            elif (key in ["from", "layers", "mask"]) or (key=='size' and ',' in value):
                mdefs[-1][key] = [int(x) for x in value.split(",")]
            else:
                if value.isnumeric():
                    mdefs[-1][key] = int(value) if (int(value) - float(value)) == 0 else float(value)
                else:
                    mdefs[-1][key] = value
    supported = ['type', 'batch_normalize', 'filters', 'size', 'stride', 'pad', 'activation', 'layers', 'groups',
                     'from', 'mask', 'anchors', 'classes', 'num', 'jitter', 'ignore_thresh', 'truth_thresh', 'random',
                     'stride_x', 'stride_y', 'weights_type', 'weights_normalization', 'scale_x_y', 'beta_nms',
                     'nms_kind','iou_loss', 'iou_normalizer', 'cls_normalizer', 'iou_thresh', 'probability']
    for x in mdefs[1:]:
        for k in x:
            if k not in supported:
                raise ValueError("Unsupported fields:{} in cfg".format(k))
    return mdefs

def dynamica_install(modules,cls_type,opt):
    '''
    动态从每个py文件中寻找对应的方法并将方法的形参传入
    :param modules: List
    :param cls_type: str
    :param opt: Dict
    :return: module
    '''
    for m in modules:
        cls_ = getattr(m,cls_type,None)
        if cls_ is not None:
            break
    if cls_ is None:
        raise ValueError(f'{cls_type} is not found')
    if isinstance(opt,dict):
        return cls_(**opt)
    elif isinstance(opt,list):
        return cls_(*opt)
    else:
        raise TypeError('The opt type is not supported')


def eff_info(model):
    '''
    打印模型的参数大小和浮点运算数
    :param model: Modules
    :return: None
    '''
    flops, params = profile(model, inputs=(torch.zeros(1, 3, 640, 640),), verbose=False)
    flops_, params_ = clever_format([flops, params], "%.3f")
    print('Model Summary: %g layers, %s parameters, %s GFLOPS' % (len(list(model.parameters())), params_, flops_))

def model_info(model, verbose=False):
    '''
    打印模型网络结构信息
    :param model:
    :param verbose:
    :return:
    '''
    # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

def parse_xml_to_dict(xml):
    """
    将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
    Args:
        xml: xml tree obtained by parsing XML file contents using lxml.etree

    Returns:
        Python dictionary holding XML contents.
    """

    if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
        return {xml.tag: xml.text}

    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)  # 递归遍历标签信息
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


# if __name__ == '__main__':
#     path = '../cfg/yolov3-spp.cfg'
#     mdefs = parse_model_config(path)