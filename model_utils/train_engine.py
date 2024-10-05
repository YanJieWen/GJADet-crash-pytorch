'''
@File: train_engine.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 3月 29, 2024
@HomePage: https://github.com/YanJieWen
'''
import os
import os.path as osp
import importlib
import random
import math
import sys
import time

import torch
from torch.cuda import amp
import torch.nn.functional as F
# import torchvision.transforms.functional as ts

from model_utils.parse_config import dynamica_install
from model_utils.distributed_utils import *
from model_utils.train_utils import computer_loss,non_max_suppression,scale_coords,normalize_img
from model_utils.coco_utils import *
from model_utils.coco_eval import CocoEvaluator



from visual_test import plot_visuals

def install_detector(backbone_name,fpn_name,nc,verbose=True):
    cfg_models = ['VGG16','Convnext','Resnet50','SwinTransformer','DLA60','Darknet','CSPDarknet']
    cfg_fpns = ['BackboneWithPAFPN','BackboneWithFPN','BackboneWithDyhead','FeaturePyramidNetwork','PAFPN','DyHead']
    assert backbone_name in cfg_models, f'The {backbone_name} is not in the builds models'
    # assert fpn_name in cfg_fpns, f'The {fpn_name} is not in the builds FPNs'
    if backbone_name=='SwinTransformer':
        assert fpn_name in cfg_fpns[-3:],f'The TF-based detector only supported FPN-ONLY'
    model_files = [osp.splitext(osp.basename(x))[0] for x in os.listdir('./create_models/') if x.startswith('Yolov3')]
    arch_libs = [importlib.import_module(f'create_models.{x}') for x in model_files]
    if backbone_name not in ['Darknet','CSPDarknet']:
        #修改本行改变anchor和预训练模型的参数
        cfg_dict = {'fpn_type': fpn_name,'verbose':verbose,'pretrained_root':'./pretrained','nc':nc}
        model = dynamica_install(arch_libs, f'{backbone_name}_Yolo', cfg_dict)
    else:
        if backbone_name == 'Darknet':
            dark_yolo_path = './cfg/yolov3-spp.cfg'
            cfg_dict = {'cfg':dark_yolo_path,'verbose':verbose,'pretrained_root':'./pretrained'}
            model = dynamica_install(arch_libs, f'{backbone_name}', cfg_dict)
        elif backbone_name == 'CSPDarknet':
            cm_dict = {'cfg_file':'./cfg/yolov8.yaml','nc':nc,'type':'m'}
            cfg_dict = {'cfg': cm_dict, 'verbose': verbose, 'pretrained_root': './pretrained'}
            model = dynamica_install(arch_libs, f'{backbone_name}', cfg_dict)
    return model

def install_dataloader(dataset_type,data_cfg,read_type='yolo'):
    data_list = ['coco','voc','cityscapes']
    assert dataset_type in data_list, f'{dataset_type} is not in the methods'
    data_files = [osp.splitext(osp.basename(x))[0] for x in os.listdir('create_datas')]
    arch_libs = [importlib.import_module(f'create_datas.{x}') for x in data_files]
    dataset = dynamica_install(arch_libs,f'{dataset_type.upper()}datasets',data_cfg)
    batch_size = data_cfg['batch_size']
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    if read_type=='yolo':
        dataloader =  torch.utils.data.DataLoader(dataset,
                                                   batch_size=batch_size,
                                                   num_workers=nw,
                                                   # Shuffle=True unless rectangular training is used
                                                   shuffle=not data_cfg['rect'],
                                                   pin_memory=True,
                                                   collate_fn=dataset.collate_fn_yolo)
    else:
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size ,
                                                 num_workers=nw,
                                                 # Shuffle=True unless rectangular training is used
                                                 shuffle=not data_cfg['rect'],
                                                 pin_memory=True,
                                                 collate_fn=dataset.collate_fn_rcnn)
    return dataloader,dataset


def install_loss_computing(losses_type,loss_cfg):
    loss_list = ['Focal','Iou','QualityFocal','EQ','GradientJointRepresentation']
    assert losses_type in loss_list,f'{losses_type} is not in the methods'
    losses_files = [osp.splitext(osp.basename(x))[0] for x in os.listdir('create_losses') if x.endswith('loss.py')]
    arch_libs = [importlib.import_module(f'create_losses.{x}') for x in losses_files]
    loss_fnc = dynamica_install(arch_libs,f'{losses_type}Loss',loss_cfg)
    return loss_fnc


def decouple_training(model,backbone_name):
    layer_index = None
    if backbone_name=='VGG16':
        for name,param in model.backbonewithfpn.named_parameters():
            if int(name.split('.')[2])<16 and name.split('.')[1]=='features':
                param.requires_grad = False
            else:
                param.requires_grad = True
    elif backbone_name=='Resnet50':
        layer_index = ['layer2', 'layer3','layer4']
    elif backbone_name=='DLA60':
        layer_index = ['level3','level4','level5']
    elif backbone_name=='Convnext':
        layer_index = ['stages.1','stages.2','stages.3']
    elif backbone_name=='SwinTransformer':
        layer_index = ['layers.1', 'layers.2', 'layers.3']
    elif backbone_name=='Darknet':
        pass
    elif backbone_name=='CSPDarknet':
        pass
    else:
        raise ValueError(f'{backbone_name} is not in the model')
    if backbone_name in ['Resnet50','DLA60','Convnext']:
        for name,parm in model.backbonewithfpn.body.named_parameters():
            if all([not name.startswith(x) for x in layer_index]):
                parm.require_grads = False
            else:
                parm.require_grads = True
    elif backbone_name=='SwinTransformer':
        for name,parm in model.backbone.named_parameters():
            if all([not name.startswith(x) for x in layer_index]):
                parm.require_grads = False
            else:
                parm.require_grads = True
    else:
        pass

def load_weights(model,optimizer,results_file,epochs,first_epochs,scaler,amp,weights,device):
    init_epochs = first_epochs
    best_map = 0.
    if weights != "":
        if weights.endswith('.pt') or weights.endswith('.pth'):
            ckpt = torch.load(weights, map_location=device)
            # load state dict
            ckpt['model'] = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(ckpt['model'], strict=False)
            if ckpt['optimizer'] is not None:
                optimizer.load_state_dict(ckpt['optimizer'])
                if 'best_map' in ckpt.keys():
                    best_map = ckpt['best_map']
            if ckpt.get("training_results") is not None:
                with open(results_file, "w") as file:
                    file.write(ckpt["training_results"])
            start_epoch = ckpt['epoch'] + 1  # 要求第一个训练阶段必须完成>
            if epochs <= start_epoch:
                print('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                          (weights, ckpt['epoch'], epochs))
                init_epochs = start_epoch
                epochs += (epochs - first_epochs)
            else:
                init_epochs = start_epoch
            if amp and "scaler" in ckpt:
                scaler.load_state_dict(ckpt["scaler"])
            del ckpt
    return init_epochs,best_map


def train_one_epoch(model,optimizer,data_loader,
                    device,epoch,accumulate,img_size,
                    multi_scale,grid_min,grid_max,gs,print_feq, warmup,scaler,obj_func,cls_func,loc_func):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    lr_scheduler = None
    if epoch == 0 and warmup is True:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
        accumulate = 1
    mloss = torch.zeros(4).to(device)
    now_lr =0.
    nb = len(data_loader)
    for i,(imgs,targets,paths,_,_) in enumerate(metric_logger.log_every(data_loader,print_feq,header)):
        ni = i+nb*epoch
        imgs = imgs.to(device).float() / 255.0
        # imgs = normalize_img(imgs)
        targets = targets.to(device)
        if multi_scale:
            if ni%accumulate==0:
                img_size = random.randrange(grid_min,grid_max+1)*gs
            sf = img_size/max(imgs.shape[2:])
            if sf!=1:
                ns = [math.ceil(x*sf/gs)*gs for x in imgs.shape[2:]]
                imgs = F.interpolate(imgs,size=ns,mode='bilinear',align_corners=False)
        with amp.autocast(enabled=scaler is not None):
            pred = model(imgs) #List[Tensor]
            loss_dict = computer_loss(pred,targets,model,obj_func,cls_func,loc_func)
            losses = sum(loss for loss in loss_dict.values())
        loss_dict_reduced =reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_items = torch.cat((loss_dict_reduced["box_loss"],
                                loss_dict_reduced["obj_loss"],
                                loss_dict_reduced["class_loss"],
                                losses_reduced)).detach()
        mloss = (mloss * i + loss_items) / (i + 1)#[l(t-1)*iter+l(t)]/(t+1)
        if not torch.isfinite(losses_reduced):
            print('WARNING: non-finite loss, ending training ', loss_dict_reduced)
            print("training image path: {}".format(paths))
            xmin = targets[:,2]-targets[:,4]/2
            ymin = targets[:,3]-targets[:,5]/2
            xmax = targets[:,2]+targets[:,4]/2
            ymax = targets[:,3]+targets[:,5]/2
            plot_visuals(imgs,targets)
            print(f'xyminmin:{[torch.min(xmin),torch.min(ymin)]}')
            print(f'xymaxmax:{[torch. max(xmax), torch.max(ymax)]}')
            sys.exit(1)
        losses *= 1. / accumulate
        if scaler is not None:
            scaler.scale(losses).backward()
        else:
            losses.backward()
        #每遍历64张图像更新一次梯度
        if ni%accumulate==0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)
        if ni % accumulate == 0 and lr_scheduler is not None:  # 第一轮使用warmup训练方式
            lr_scheduler.step()
    return mloss,now_lr
@torch.no_grad()
def evaluate(model,data_loader,coco,device):
    cpu_device = torch.device('cpu')
    model.eval()
    metric_logger = MetricLogger(delimiter=" ")
    header ="Test"
    if coco is None:
        coco = get_coco_api_from_dataset(data_loader.dataset)#将gt转为coco数据集
    iou_types =_get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco,iou_types)#COCO,LIST[DEFUALT=BBOX]，gt将coco坐标转为xyxy格式，coco类中包含了数据的原始信息
    for imgs,targets,paths,shapes,img_index in metric_logger.log_every(data_loader,100,header):
        imgs = imgs.to(device).float()/255.0
        # imgs = normalize_img(imgs)
        if device!=torch.device('cpu'):#if gpu ignored command from gpu
            torch.cuda.synchronize(device)
        model_time = time.time()
        pred,_ = model(imgs) #[N,numberanchors,85]
        #后处理
        pred = non_max_suppression(pred,conf_thres=0.01,iou_thres=0.6,multi_label=False)
        model_time = time.time()-model_time

        #尺度缩放回原图
        outputs = []
        for index,p in enumerate(pred):#遍历每一张图像
            if p is None:
                p = torch.empty((0,6),device=cpu_device)
                boxes = torch.empty((0,4),device=cpu_device)
            else:

                boxes = p[:, :4]#xyxy
                boxes = scale_coords(imgs[index].shape[1:], boxes, shapes[index][0]).round()
            info = {"boxes": boxes.to(cpu_device),
                    "labels": p[:, 5].to(device=cpu_device, dtype=torch.int64)+1,#因为gt是从1开始计数
                    "scores": p[:, 4].to(cpu_device)}
            outputs.append(info)
        res = {img_id: output for img_id, output in zip(img_index, outputs)}#仅计算annotations的mAP
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    result_info = coco_evaluator.coco_eval[iou_types[0]].stats.tolist()  # numpy to list

    return result_info #返回list

def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    return iou_types


