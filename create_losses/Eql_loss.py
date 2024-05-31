'''
@File: Eql_loss.py
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
from functools import partial

import torch.distributed as dist

class EQLoss(nn.Module):
    def __init__(self,use_sigmoid=True,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 num_classes=20,
                 gamma=12,
                 mu=0.8,
                 alpha=4.0,
                 vis_grad=False,
                 test_with_obj=True):
        super(EQLoss,self).__init__()
        self.use_sigmoid =  True
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.num_classes = num_classes
        self.group = True
        #for eq
        self.vis_grad = vis_grad
        self.gamma = gamma
        self.mu = mu
        self.alpha = alpha
        self.register_buffer('pos_grad',torch.zeros(self.num_classes,device='cuda:0'))#c
        self.register_buffer('neg_grad',torch.zeros(self.num_classes,device='cuda:0'))#c
        self.register_buffer('pos_neg',torch.ones(self.num_classes,device='cuda:0')*100)#c

        def _func(x,gamma,mu):
            return 1/(1+torch.exp(-gamma*(x-mu)))
        self.map_func = partial(_func,gamma=self.gamma,mu=self.mu)

    def forward(self,pred,label,weight=None,avg_factor=None,reduction_override=None,**kwargs):
        '''
        基于梯度引导的均衡损失函数--目前暂时仅用于类别损失
        :param pred:筛选后的n,ci
        :param label:筛选后的n,ci---one hot编码
        :param weight:None
        :param avg_factor:None
        :param reduction_override:None
        :param kwargs:
        :return:
        '''
        self.n_i,self.n_c =  pred.size()
        self.gt_classes =  label
        self.pred_class_logits = pred
        # self.pos_grad.to(pred.device)
        # self.neg_grad.to(pred.device)
        # self.pos_neg.to(pred.device)
        #根据预测值拓展维度
        pos_w,neg_w = self.get_weight(pred)
        #对t时刻的梯度进行加权
        weight = pos_w*label+neg_w*(1-label)#根据gj更新权值
        cls_loss =  F.binary_cross_entropy_with_logits(pred,label,reduction='none')#这个地方的改进可以使用qfl
        cls_loss =  torch.sum(cls_loss*weight)/self.n_i#权值作为调制因子进行矫正
        self.collect_grad(pred.detach(),label.detach(),weight.detach())#收集类别损失梯度

        return self.loss_weight*cls_loss


    def collect_grad(self,pred,target,weight):
        '''
        更新t+1时刻的梯度
        :param pred:n,c
        :param target:n,c
        :param weight:n,c
        :return:
        '''
        prob = torch.sigmoid(pred)
        grad = target*(prob-1)+(1-target)*prob
        grad = torch.abs(grad)
        pos_grad = torch.sum(grad*target*weight,dim=0)#整梯度综合
        neg_grad = torch.sum(grad*(1-target)*weight,dim=0)
        #
        # dist.all_reduce(pos_grad)
        # dist.all_reduce(neg_grad)

        self.pos_grad += pos_grad
        self.neg_grad += neg_grad

        self.pos_neg = self.pos_grad / (self.neg_grad + 1e-10)#更新gj



    def get_weight(self,cls_score):
        '''
        获得正负样本的梯度加权值
        :param cls_score:
        :return:
        '''
        # neg_w = torch.cat([self.map_func(self.pos_neg),cls_score.new_ones(1)])#c，加1是因为多出一个背景属性
        neg_w = self.map_func(self.pos_neg.to(cls_score.device))
        pos_w = 1+self.alpha*(1-neg_w)#c
        neg_w = neg_w.view(1,-1).expand(self.n_i,self.n_c)
        pos_w = pos_w.view(1,-1).expand(self.n_i,self.n_c)
        return pos_w,neg_w

