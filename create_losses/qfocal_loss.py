'''
@File: qfocal_loss.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 4月 01, 2024
@HomePage: https://github.com/YanJieWen
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss_utils import *

def quality_focal_loss(pred,target,beta=2.0,weight=None,reduction='mean',avg_factor=None):
    pt = torch.sigmoid(pred)
    modulator = torch.abs(target-pt).pow(beta)
    loss = F.binary_cross_entropy_with_logits(pred,target,reduction='none')*modulator
    loss = weight_reduce_loss(loss,weight,reduction,avg_factor)
    return loss

class QualityFocalLoss(nn.Module):
    def __init__(self,use_sigmoid=True,beta=2.0,reduction='mean',loss_weight=1.0):
        super(QualityFocalLoss,self).__init__()
        assert use_sigmoid is True,'only sigmoid in QFL surpported now.'
        self.use_sigmpid = use_sigmoid
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,pred,target,weight=None,avg_factor=None,reduction_override=None):
        '''
        质量-分类联合表征损失
        :param pred: [n,c]
        :param target:n
        :param weight:None
        :param avg_factor:None
        :param reduction_override:None
        :return:
        '''
        assert reduction_override in (None,'none','mean','sum')
        reduction = (reduction_override if reduction_override else self.reduction)
        if self.use_sigmpid:
            loss_cls = self.loss_weight * quality_focal_loss(
                pred,
                target,
                weight=weight,
                beta = self.beta,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls
