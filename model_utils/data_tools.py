'''
@File: data_tools.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 3月 28, 2024
@HomePage: https://github.com/YanJieWen
'''
import copy
import random

import torch
import torch.utils.data
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

import torchvision.transforms as ts

import os
import os.path as osp
import cv2
import numpy as np
import math




def coco_remove_images_without_annotations(dataset, ids):
    """
    删除coco数据集中没有目标，或者目标面积非常小的数据
    refer to:
    https://github.com/pytorch/vision/blob/master/references/detection/coco_utils.py
    :param dataset:
    :param cat_list:
    :return:
    """
    def _has_only_empty_bbox(anno):
        return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

    def _has_valid_annotation(anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if all boxes have close to zero area, there is no annotation
        if _has_only_empty_bbox(anno):
            return False

        return True

    valid_ids = []
    for ds_idx, img_id in enumerate(ids):
        ann_ids = dataset.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.loadAnns(ann_ids)

        if _has_valid_annotation(anno):
            valid_ids.append(img_id)

    return valid_ids



def convert_coco_poly_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        # 如果mask为空，则说明没有目标，直接返回数值为0的mask
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def load_image(self,index):
    '''
    加载图像
    :param self: dataclass
    :param index: 索引
    :return: 按照最长边的等比例缩放到指定尺寸
    '''
    coco = self.coco
    img_id = self.ids[index]
    img_info = coco.loadImgs(img_id)[0]
    img_file = img_info['file_name']
    img_path = osp.join(self.img_root, img_file)
    img = cv2.imread(img_path)
    assert img is not None, "Image Not Found " + img_path
    h0, w0 = img.shape[:2]
    r = self.img_size / max(h0, w0)
    if r != 1:
        interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
    return img, (h0, w0), img.shape[:2]

def load_mosaic(self,index,transforms):
    '''
    马赛克增强
    :param self: 数据加载类
    :param index: 图像的id
    :param transforms:增强手段
    :return:马赛克增强图和tgt
    '''
    assert len(transforms)==4, 'the number of augers is equal to 4'
    coco = self.coco
    s = self.img_size
    new_targets = {}
    bboxes = []
    classes = []
    areas = []
    iscrowds = []
    xc,yc =[int(random.uniform(s*0.5,s*1.5)) for _ in range(2)]
    indices = [index]+[random.randint(0, len(self.ids) - 1) for _ in range(3)]
    for i,index in enumerate(indices):
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_targets = coco.loadAnns(ann_ids)
        img_info = coco.loadImgs(img_id)[0]
        w, h = img_info['width'], img_info['height']
        x = self.parse_target(img_id,coco_targets,w,h)
        labels = copy.deepcopy(x) #dict
        img,_,(h,w) = load_image(self,index)
        img,labels['boxes'] = transforms[i](img,labels['boxes'].numpy())
        if i == 0:
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif i == 1:
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
        elif i == 3:
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
        # 计算pad(图像边界与马赛克边界的距离，越界的情况为负值)
        padw = x1a - x1b
        padh = y1a - y1b
        if x['boxes'].size()[0]>0:
            if self.model_type=='yolo':
                labels['boxes'][:,0] = w*(x['boxes'][:,0]-x['boxes'][:,2]/2)+padw #xmin
                labels['boxes'][:,1] = h*(x['boxes'][:, 1] - x['boxes'][:, 3] / 2) + padh #ymin
                labels['boxes'][:,2] = w * (x['boxes'][:, 0] + x['boxes'][:, 2] / 2) + padw#xmax
                labels['boxes'][:,3] = h * (x['boxes'][:, 1] + x['boxes'][:, 3] / 2) + padh#ymax
            else:#fasterrcnn
                labels['boxes'][:,0] = w*x['boxes'][:,0]+padw
                labels['boxes'][:, 1] = h * x['boxes'][:, 1] + padh
                labels['boxes'][:, 2] = w * x['boxes'][:, 2] + padw
                labels['boxes'][:, 3] = h * x['boxes'][:, 3] + padh
            bboxes.append(labels['boxes'])
            classes.extend(labels['classes'])
            areas.extend(labels['area'])
            iscrowds.extend(labels['iscrowd'])
    if len(bboxes):
        bboxes = torch.as_tensor(np.concatenate(bboxes,0))#均转为左上和右下角坐标
        torch.clamp_(bboxes,0,2*s)
        classes = torch.as_tensor(classes).reshape(-1,1)
        areas = torch.as_tensor(areas).reshape(-1,1)
        iscrowds = torch.as_tensor(iscrowds).reshape(-1,1)
    bboxes = torch.concatenate((classes,areas,iscrowds,bboxes),dim=-1).numpy()#[N,5]
    img4, bboxes = random_affine(img4, bboxes,
                                  degrees=0.,
                                  translate=0.,
                                  scale=0.,
                                  shear=0.,
                                  border=-s // 2) #防止坐标越界
    bboxes = torch.as_tensor(bboxes)
    new_targets['boxes'] = bboxes[:,3:7]
    new_targets['classes'] = bboxes[:,0]
    new_targets['area'] = bboxes[:,1]
    new_targets['iscrowd'] = bboxes[:,2]

    return img4,new_targets







def random_affine(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, border=0):
    """随机旋转，缩放，平移以及错切"""
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
    # 这里可以参考我写的博文: https://blog.csdn.net/qq_37541097/article/details/119420860
    # targets = [cls, xyxy]

    # 最终输出的图像尺寸，等于img4.shape / 2->缩放到原始尺寸
    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2

    # Rotation and Scale
    # 生成旋转以及缩放矩阵
    R = np.eye(3)  # 生成对角阵
    a = random.uniform(-degrees, degrees)  # 随机旋转角度
    s = random.uniform(1 - scale, 1 + scale)  # 随机缩放因子
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    # 生成平移矩阵
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border  # y translation (pixels)

    # Shear
    # 生成错切矩阵
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Combined rotation matrix
    M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
    if (border != 0) or (M != np.eye(3)).any():  # image changed
        # 进行仿射变化
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [3,4,5,6, 3,6,5,4]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        # [4*n, 3] -> [n, 8]
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # create new boxes
        # 对transform后的bbox进行修正(假设变换后的bbox变成了菱形，此时要修正成矩形)
        x = xy[:, [0, 2, 4, 6]]  # [n, 4]
        y = xy[:, [1, 3, 5, 7]]  # [n, 4]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T  # [n, 4]

        # reject warped points outside of image
        # 对坐标进行裁剪，防止越界
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]

        # 计算调整后的每个box的面积
        area = w * h
        # 计算调整前的每个box的面积
        area0 = (targets[:, 5] - targets[:, 3]) * (targets[:, 6] - targets[:, 4])
        # 计算每个box的比例
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
        # 选取长宽大于4个像素，且调整前后面积比例大于0.2，且比例小于10的box
        i = (w > 2) & (h > 2) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 10)

        targets = targets[i]
        targets[:, 3:7] = xy[i]

    return img, targets



def letterbox(img: np.ndarray,
              new_shape=(416, 416),
              color=(114, 114, 114),
              auto=True,
              scale_fill=False,
              scale_up=True):
    """
    将图片缩放调整到指定大小
    :param img:
    :param new_shape:
    :param color:
    :param auto:
    :param scale_fill:
    :param scale_up:
    :return:
    """

    shape = img.shape[:2]  # [h, w]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scale_up:  # only scale down, do not scale up (for better test mAP) 对于大于指定输入大小的图片进行缩放,小于的不变
        r = min(r, 1.0)

    # compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimun rectangle 保证原图比例不变，将图像最大边缩放到指定大小
        # 这里的取余操作可以保证padding后的图片是32的整数倍
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scale_fill:  # stretch 简单粗暴的将图片缩放到指定尺寸
        dw, dh = 0, 0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # wh ratios

    dw /= 2  # divide padding into 2 sides 将padding分到上下，左右两侧
    dh /= 2

    # shape:[h, w]  new_unpad:[w, h]
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))  # 计算上下两侧的padding
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))  # 计算左右两侧的padding

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y
