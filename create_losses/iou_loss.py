'''
@File: iou_loss.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 3月 31, 2024
@HomePage: https://github.com/YanJieWen
'''

import torch.nn as nn

import torch
import math

class IouLoss(nn.Module):
    def __init__(self,x1y1x2y2=True,GIoU=False,DIoU=False,CIoU=False):
        super(IouLoss,self).__init__()
        self.x1y1x2y2 = x1y1x2y2
        self.giou = GIoU
        self.diou = DIoU
        self.ciou = CIoU

    def forward(self,pred,tgt):
        '''
        基于IOU计算损失函数
        :param pred: Tensor[4,n]
        :param tgt: Tensor[n,4]
        :return: iou tensor[N]
        '''
        box2 = tgt.t()

        # Get the coordinates of bounding boxes
        if self.x1y1x2y2:  # x1, y1, x2, y2 = box1
            b1_x1, b1_y1, b1_x2, b1_y2 = pred[0], pred[1], pred[2], pred[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
        else:  # transform from xywh to xyxy
            b1_x1, b1_x2 = pred[0] - pred[2] / 2, pred[0] + pred[2] / 2
            b1_y1, b1_y2 = pred[1] - pred[3] / 2, pred[1] + pred[3] / 2
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
        if self.giou or self.diou or self.ciou:
            cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
            ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
            if self.giou:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
                c_area = cw * ch + 1e-16  # convex area
                return iou - (c_area - union) / c_area  # GIoU
            if self.diou or self.ciou:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
                # convex diagonal squared
                c2 = cw ** 2 + ch ** 2 + 1e-16
                # centerpoint distance squared
                rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
                if self.diou:
                    return iou - rho2 / c2  # DIoU
                elif self.ciou:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                    v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                    with torch.no_grad():
                        alpha = v / (1 - iou + v)
                    return iou - (rho2 / c2 + v * alpha)  # CIoU
        return iou
