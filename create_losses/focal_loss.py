'''
@File: focal_loss.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 3月 31, 2024
@HomePage: https://github.com/YanJieWen
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss_utils import *


def sigmoid_focal_loss(pred,target,weight=None,loss_func=None,gamma=2.0,alpha=0.25,reduction='mean',avg_factor=None):
    pred_sigmoid = torch.sigmoid(pred)
    target = target.type_as(pred)
    pt = pred_sigmoid*target+(1-pred_sigmoid)*(1-target)
    if gamma==0.:#回到二值交叉熵损失
        focal_weight = 1.
    else:
        focal_weight = (alpha*target+(1-alpha)*(1-target))*((1-pt)**gamma)#用target来区分正负样本
    loss = F.binary_cross_entropy_with_logits(pred,target,reduction='none')*focal_weight
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


class FocalLoss(nn.Module):
    def __init__(self,use_sigmoid=True,gamma=2.0,alpha=0.25,reduction='mean',loss_weight=1.0):
        super(FocalLoss,self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,pred,target,weight=None,avg_factor=None,reduction_override=None):
        '''
        Focal Loss <https://arxiv.org/abs/1708.02002>
        :param pred:[N,C]
        :param target:[N,C]->[0,1]
        :param weight:The weight of loss for each prediction. Defaults to None
        :param avg_factor:Average factor that is used to average the loss. Defaults to None
        :param reduction_override:The reduction method used to  override the original
        reduction method of the loss. Options are "none", "mean" and "sum"
        :return:
        '''
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss_cls = self.loss_weight * sigmoid_focal_loss(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls