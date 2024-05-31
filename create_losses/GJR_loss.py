'''
@File: GJRLoss.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 4月 02, 2024
@HomePage: https://github.com/YanJieWen
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class GradientJointRepresentationLoss(nn.Module):
    def __init__(self,num_classes=80,
                 gamma=12,mu=0.8,alpha=4.0,sigma=2.0):
        super(GradientJointRepresentationLoss,self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.mu = mu
        self.alpha = alpha
        self.sigma = sigma
        self.register_buffer('pos_grad', torch.zeros(self.num_classes, device='cuda:0'))  # c
        self.register_buffer('neg_grad', torch.zeros(self.num_classes, device='cuda:0'))  # c
        self.register_buffer('pos_neg', torch.ones(self.num_classes, device='cuda:0') * 100)  # c
        def _func(x,gamma,mu):
            return 1/(1+torch.exp(-gamma*(x-mu)))
        self.map_func = partial(_func,gamma=self.gamma,mu=self.mu)

    def forward(self,pred,target,weight=None,avg_factor=None,reduction_override=None):
        self.n_i, self.n_c = pred.size()
        self.gt_classes = target
        self.pred_class_logits = pred
        pos_w,neg_w = self.get_weight(pred)
        weight = pos_w*target+(1-target)*neg_w
        # gamma_b = neg_w-self.alpha*(1-self.pos_neg)
        # gamma_v = 1-self.pos_neg
        # gamme_w = (gamma_b+gamma_v)/gamma_b
        #计算联合表征
        pt = torch.sigmoid(pred)
        modulator = torch.abs(target-pt).pow(self.beta)
        # modulator = torch.abs(target-pt).pow(self.pos_neg)*gamme_w
        cls_loss = F.binary_cross_entropy_with_logits(pred,target,reduction='none')*modulator
        cls_loss = torch.sum(cls_loss * weight) / self.n_i
        self.collect_grad(pred.detach(),target.detach(),weight.detach())
        return cls_loss


    def collect_grad(self, pred, target, weight):
        '''
        更新t+1时刻的梯度
        :param pred:n,c
        :param target:n,c
        :param weight:n,c
        :return:
        '''
        prob = torch.sigmoid(pred)
        grad = target * (prob - 1) + (1 - target) * prob
        grad = torch.abs(grad)
        pos_grad = torch.sum(grad * target * weight, dim=0)  # 整梯度综合
        neg_grad = torch.sum(grad * (1 - target) * weight, dim=0)
        #
        # dist.all_reduce(pos_grad)
        # dist.all_reduce(neg_grad)

        self.pos_grad += pos_grad
        self.neg_grad += neg_grad

        self.pos_neg = self.pos_grad / (self.neg_grad + 1e-10)  # 更新gj

    def get_weight(self, cls_score):
        '''
        获得正负样本的梯度加权值
        :param cls_score:
        :return:
        '''
        # neg_w = torch.cat([self.map_func(self.pos_neg),cls_score.new_ones(1)])#c，加1是因为多出一个背景属性
        neg_w = self.map_func(self.pos_neg.to(cls_score.device))
        pos_w = 1 + self.alpha * (1 - neg_w)  # c
        neg_w = neg_w.view(1, -1).expand(self.n_i, self.n_c)
        pos_w = pos_w.view(1, -1).expand(self.n_i, self.n_c)
        return pos_w, neg_w