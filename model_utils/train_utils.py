'''
@File: train_utils.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 3月 29, 2024
@HomePage: https://github.com/YanJieWen
'''

import torch
import torch.nn as nn
import torchvision

import math
import numpy as np
import time

def computer_loss(p,tgts,model,obj_func,cls_func,loc_func):
    '''
    计算损失
    :param p:List[Tensor]底层-->顶层
    :param tgts:Tensor[N,6]
    :param model: module
    :return:dict-->loss
    '''
    assert isinstance(p,list), 'the framework is not multi-scale detector'
    device = p[0].device
    lcls = torch.zeros(1,device=device)
    lbox = torch.zeros(1,device=device)
    lobj = torch.zeros(1,device=device)
    cls, tbox, indices, anch = samplers(p,tgts,model)
    # red = 'mean'
    # BCEcls = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0],device=device),reduction=red)
    # BCEobj = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0],device=device),reduction=red)
    # cp,cn = 1.0, 0.0
    # fl_gamma = 0. #focal_loss
    # if fl_gamma>0:
    #     BCEcls, BCEobj = FocalLoss(BCEcls, fl_gamma), FocalLoss(BCEobj, fl_gamma)

    for i,pi in enumerate(p):
        b,a,gj,gi = indices[i] #image_id,anchor_idx,grid_y,grid_x
        tobj = torch.zeros_like(pi[...,0],device=device) #obj标签
        nb = b.shape[0]
        if nb:
            ps = pi[b,a,gj,gi]#所有正样本的预测信息[n,85]
            pxy = ps[:,:2].sigmoid()#偏移量
            pwh = torch.exp(ps[:,2:4]).clamp(max=1E3)*anch[i]
            pbox = torch.cat((pxy,pwh),dim=1)
            # iou = bbox_iou(pbox.t(), tbox[i], x1y1x2y2=False,GIoU=True)
            iou = loc_func(pbox.t(), tbox[i])
            lbox+=(1.0-iou).mean()
            #obj真值
            tobj[b,a,gj,gi] = iou.detach().clamp(0).type(tobj.dtype)#保证标签为0-1


            if model.nc>1:
                t = torch.full_like(ps[:,5:],0,device=device)#[n,80]-->这里的cls是从1开始排序
                if model.hp['cls_name'] in ['QualityFocal','GradientJointRepresentation']:
                    t[range(nb), cls[i] - 1] = iou.detach().clamp(0).type(tobj.dtype)#因为类的编码是从1开始所以需要-1进行索引,联合表示
                else:
                    t[range(nb), cls[i] - 1] = 1.0
                # lcls+=BCEcls(ps[:,5:],t) #仅计算正样本的分类情况
                lcls += cls_func(ps[:,5:],t)
        # lobj+=BCEobj(pi[...,4],tobj)
        lobj += obj_func(pi[...,4],tobj)
    lbox*=model.hp['loc_weight']
    lobj*=model.hp['obj_weight']
    lcls*=model.hp['cls_weight']
    return {'box_loss':lbox,'obj_loss':lobj,'class_loss':lcls}




def samplers(pred,targets,model):
    '''
    进行正样本采样
    :param pred: List[Tensor]-->从小到大的特征图排序
    :param targets: Tensor[N,6]
    :param model:module
    :return: 对于每一个特征层：indices[imid,anchorid,gj,gi],tbox[tx,ty,w,h],anch[筛选的正样本anchors索引]，tcls分类，
    '''
    nt = targets.shape[0]
    tcls,tbox,indices,anch = [],[],[],[]
    gain = torch.ones(6,device=targets.device).long()
    multi_gpu = type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    names = ['backbone' in name for name,_ in model.named_modules()]
    try:
        if np.any(names):
            all_anchors = [yolo_layer.anchor_vec.to(pred[0].device) for yolo_layer in model.yolo_layers]
            all_anchors.reverse()
        else:
            all_anchors = [model.module.module_list[j].anchor_vec.to(pred[0].device) for j in model.yolo_layers] if multi_gpu else [
                model.module_list[j].anchor_vec.to(pred[0].device) for j in model.yolo_layers]
    except Exception as e: #如果是CSPDarknet
        all_anchors = [detect.anchor_vec.to(pred[0].device) for detect in model.yolo_layers]
        all_anchors.reverse()

    if all_anchors is None:
        raise ValueError('backbone is not in the model')

    for i,anchors in enumerate(all_anchors):
        gain[2:] = torch.tensor(pred[i].shape)[[3,2,3,2]]#(whwh)
        na = anchors.shape[0]
        at = torch.arange(na,device=targets.device).view(na,1).repeat(1,nt) #[na,nt]一个anchor匹配多个gt，一个gt也可能匹配多个anchor
        a,t,offsets = [],targets*gain,0 #tgt*gain获得anchors在该特征图尺寸下的大小
        if nt:
            j = bbox_assiner(anchors,t)
            a,t = at[j],t.repeat(na,1,1)[j]#[n],[n,6]

        b,c = t[:,:2].long().T
        gxy = t[:,2:4]
        gwh = t[:,4:6]
        gij = (gxy-offsets).long()#long取整数表示tgt所在的gridcell左上角坐标
        gi,gj = gij.T

        indices.append((b,a,gj.clamp_(0,gain[3]-1),gi.clamp_(0,gain[2]-1)))#正样本对应的imgindex，正样本对应的anchor模板，gj，gi为正样本中心点坐标
        tbox.append(torch.cat((gxy - gij,gwh),dim=1)) #(真值中心点坐标相对于左上角坐标的偏移量，gwh)
        anch.append(anchors[a])#正样本的anchor模板索引
        tcls.append(c)
        if c.shape[0]:
            assert c.max()<=model.nc, 'model class has been over-sized'

    return tcls,tbox,indices,anch




def bbox_assiner(anchors,tgts):
    '''
    基于anchors的模板匹配方法
    :param anchors:[3,2]
    :param tgts: [N,6]
    :return: 正样本索引j-->3,n
    '''
    tgt = tgts[:,4:6]#[n,2]
    j = templete_iou(anchors,tgt)>0.2
    return j


def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)

def templete_iou(wh1,wh2):
    '''
    基于iou的模板匹配
    :param wh1: anchors
    :param wh2: tgts
    :return: Tensor[3,n]
    '''
    wh1 = wh1[:, None]  # 3,1,2
    wh2 = wh2[None]  # 1,n,2
    inter = torch.min(wh1,wh2).prod(2)
    return inter/(wh1.prod(2)+wh2.prod(2)-inter)



def non_max_suppression(predictions,conf_thres=0.1,iou_thres=0.6,multi_label=True,classes=None,agnostic=False,max_num=200):
    '''
    merge-nms：加权下的非极大值抑制
    :param predictions: [n,xywhocls]
    :param conf_thres: float
    :param iou_thres: float
    :param multi_label: bool
    :param classes: None
    :param agnostic: bool
    :param max_num: int
    :return: List[Tensor(M,6)-->[xyxy,conf,cls_index from 0]]
    '''
    merge=True
    min_wh,max_wh = 2,4096
    time_limit = 10.0
    t = time.time()
    nc = predictions.shape[-1]-5
    multi_label&=nc>1
    output = [None]*predictions.shape[0]
    for xi,x in enumerate(predictions):#遍历每张图片
        x = x[x[:,4]>conf_thres]#滤除背景目标
       #滤除小目标和极大尺寸的预测值
        x = x[((x[:, 2:4] > min_wh) & (x[:, 2:4] < max_wh)).all(1)]
        if not x.shape[0]:
            continue

        x[...,5:]*=x[...,4:5] #conf = conf*cls_conf
        box = xywh2xyxy(x[...,:4])
        if multi_label: #针对每个类别进行nms
            i,j = (x[:,5:]>conf_thres).nonzero(as_tuple=False).t()#i为preds索引，j为类别索引
            #带有6个属性的pred_box[x,y,x,y,obk&cls_conf,cls_index],需要注意index是从0开始，而输入数据从1开始
            x = torch.cat((box[i],x[i,j+5].unsqueeze(1),j.float().unsqueeze(1)),dim=1)#[n,6]
        else:#直接对每个类别中概率最大的类进行nms
            conf,j = x[:,5:].max(1)#压缩列conf最大的值，j对比每一行保存的列索引
            x = torch.cat((box,conf.unsqueeze(1),j.float().unsqueeze(1)),dim=1)[conf>conf_thres]#[n,6]

        if classes:
            x = x[(j.view(-1,1)==torch.tensor(classes,device=j.device)).any(1)]

        n = x.shape[0]
        if not n:
            continue
        #batched_nms
        c = x[:,5]*0 if agnostic else x[:,5]#如果是不可知论则不关心类别数目
        boxes,scores = x[:,:4].clone()+c.view(-1,1)*max_wh,x[:,4]#根据类乘上一个最大值保持在不同类上进行nms
        i = torchvision.ops.nms(boxes,scores,iou_thres)#batch_nms
        i = i[:max_num]
        if merge and (1<n<3e3): #为boxes赋权,
            try:
                iou = box_iou(boxes[i],boxes)>iou_thres#m(nms后)xn(nms前)
                weights = iou*scores[None]#矩阵质量和得分赋权
                x[i,:4] = torch.mm(weights,x[:,:4]).float()/weights.sum(1,keepdim=True)
            except:
                print(x,i,x.shape,i.shape)
                pass

        output[xi] = x[i]
        if (time.time()-t)>time_limit:
            break

    return output


def xywh2xyxy(box):
    '''
    中心坐标转左上角和右下角坐标
    :param box: [N,4]
    :return: [N,4]
    '''
    y = torch.zeros_like(box) if isinstance(box,torch.Tensor) else np.zeros_like(box)
    y[:,0] = box[:,0]-box[:,2]/2
    y[:,1] = box[:,1]-box[:,3]/2
    y[:,2] = box[:,0]+box[:,2]/2
    y[:,3] = box[:,1]+box[:,3]/2
    return y

def box_iou(box1, box2):
    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])
    area1 = box_area(box1.t())
    area2 = box_area(box2.t())
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)




def scale_coords(img1_shape,coords,img0_shape,ratio_pad=None):
    """
     将预测的坐标信息转换回原图尺度
     :param img1_shape: 缩放后的图像尺度
     :param coords: 预测的box信息
     :param img0_shape: 缩放前的图像尺度
     :param ratio_pad: 缩放过程中的缩放比例以及pad
     :return:
     """
    if ratio_pad is None:
        gain = max(img1_shape)/max(img0_shape)#old/new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2#上下填充
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    coords[:,[0,2]] -= pad[0]#
    coords[:,[1,3]] -= pad[1]
    coords[:,:4] /= gain
    clip_coords(coords,img0_shape)#限制边界
    return coords

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


class FocalLoss(nn.Module):
    def __init__(self,loss_func,gamma=1.5,alpha=0.25):
        super(FocalLoss,self).__init__()
        self.loss_fnc = loss_func
        self.gamma = gamma
        self.alpha = alpha
        self.reducation = loss_func.reduction
        self.loss_fnc.reduction = 'none'
    def forward(self,pred,tgt):
        loss = self.loss_fnc(pred,tgt)
        pred_prob = torch.sigmoid(pred)
        p_t = tgt*pred_prob+(1-tgt)*(1-pred_prob)
        alpha_factor = tgt*self.alpha+(1-tgt)*(1-self.alpha)
        modulating_factor = (1-p_t)**self.gamma
        loss*=alpha_factor*modulating_factor
        if self.reducation =='mean':
            return loss.mean()
        elif self.reducation=='sum':
            return loss.sum()
        else:
            return loss



def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou

def normalize_img(imgs):
    dtype,device = imgs.dtype,imgs.device
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    mean = torch.as_tensor(image_mean,dtype=dtype,device=device)
    std = torch.as_tensor(image_std,dtype=dtype,device=device)
    return (imgs-mean[None,:,None,None])/std[None,:,None,None]