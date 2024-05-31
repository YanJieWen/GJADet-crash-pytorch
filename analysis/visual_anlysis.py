'''
@File: visual_anlysis.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 4月 16, 2024
@HomePage: https://github.com/YanJieWen
'''
import sys

from create_datas import Coco_datasets
import numpy as np
import os
import os.path as osp
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import PIL.ImageDraw as ImageDraw
import time

from model_utils.visual_utils import draw_objs
from model_utils.train_utils import bbox_iou




def visual_iou(gt,pred):
    gt_area = (gt[:,2]-gt[:,0])*(gt[:,3]-gt[:,1])
    pred_area = (pred[:,2]-pred[:,0])*(pred[:,3]-pred[:,1])
    lt = np.maximum(gt[:,None,:2],pred[:,:2])
    rb = np.minimum(gt[:,None,2:],pred[:,2:])
    wh = (rb-lt).clip(0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    iou = inter / (gt_area[:, None] + pred_area - inter)
    return iou

def draw_agnostic(img,boxes,color=(239, 35, 60),line_thicknes=8):
    draw = ImageDraw.Draw(img)
    for box in boxes:
        left,top,right,bottom = box
        draw.line([(left, top), (left, bottom), (right, bottom),
                       (right, top), (left, top)], width=line_thicknes, fill=color)
    return img


data_root = '../datasets/POCOCO'
data_type = 'test'
img_size = 512
pred_path = './pred_DLA60.npy'
assert osp.isfile(pred_path),f'the {pred_path} is not exists'
output_root = './run/DLA/'
if not osp.exists(output_root):
    os.makedirs(output_root)

test_datasets = Coco_datasets.COCOdatasets(data_root,data_type,batch_size=1,img_size=img_size,rect=True)
predictions = np.load(pred_path,allow_pickle=True)
coco = test_datasets.coco
coco_classes = dict([(str(v['id']),v['name']) for k,v in coco.cats.items()])
image_root = osp.join(data_root+os.sep,data_type)
img_ids = list(set(predictions[:,0]))

pbar = tqdm(img_ids,desc='Visualization images...',file=sys.stdout)
for id in pbar:
    img_path = osp.join(image_root, f'{str(int(id)).zfill(6)}.jpg')
    gt_ann_ids = coco.getAnnIds(id)
    tgts = coco.loadAnns(gt_ann_ids)
    tgt_locs = np.concatenate([[v for k, v in label.items() if k == 'bbox'] for label in tgts], axis=0)
    tgt_cats = np.concatenate([[v for k, v in label.items() if k == 'category_id'] for label in tgts], axis=0).reshape(
        -1, 1)
    tgts = np.concatenate((tgt_locs, tgt_cats), axis=1)
    preds = predictions[predictions[:, 0] == id, 1:]
    tgts[:, 2] = tgts[:, 0] + tgts[:, 2]
    tgts[:, 3] = tgts[:, 1] + tgts[:, 3]
    img = cv2.imread(img_path)[:, :, ::-1]
    img = Image.fromarray(img)
    tgt_ps = tgts[tgts[:, -1] == 1, :4]
    pred_ps = preds[preds[:, 4] == 1, :4]
    if len(pred_ps) != 0:
        # 计算IOU损失
        tgt_pred_cost = visual_iou(tgt_ps, pred_ps)
    else:
        continue
    iou_theresh = 0.5
    tp_pred_ids = np.argmax(tgt_pred_cost, axis=1)
    tp_gt_ids = np.argmax(tgt_pred_cost, axis=0)
    gt_matches = set([(i, idx) for i, idx in enumerate(tp_pred_ids)])
    pred_matches = set([(idx, i) for i, idx in enumerate(tp_gt_ids)])
    mathes_tuple = list(gt_matches.intersection(pred_matches))
    matches_ids = np.array([[idx[0], idx[1]] for idx in mathes_tuple])
    tp_dual_ids = np.where(tgt_pred_cost[matches_ids[:, 0], matches_ids[:, 1]] > iou_theresh)[0]  # 高于0.5的pred索引
    if len(tp_dual_ids) != 0:
        _tp_ids = matches_ids[tp_dual_ids][:, 1]
        _fp_ids = list(set(list(np.arange(pred_ps.shape[0]))) - set(_tp_ids))  # 误检的pred索引
        _fn_ids = list(set(list(np.arange(tgt_ps.shape[0]))) - set(matches_ids[tp_dual_ids][:, 0]))  # 漏检的索引
        tp_boxes = pred_ps[_tp_ids]
        fp_boxes = pred_ps[_fp_ids]
        fn_boxes = tgt_ps[_fn_ids]
        img = draw_agnostic(img, tp_boxes, color=(239, 35, 60), line_thicknes=3)  # 红色
        img = draw_agnostic(img, fp_boxes, color=(58, 134, 255), line_thicknes=3)  # 误检-蓝色
        img = draw_agnostic(img, fn_boxes, color=(42, 157, 143), line_thicknes=3)  # 漏检-绿色
    else:
        pbar.desc = f'{str(int(id)).zfill(6)} no any point class target'
        continue
    img = np.array(img)[:,:,::-1]
    save_path = osp.join(output_root,f'{str(int(id)).zfill(6)}.jpg')
    cv2.imwrite(save_path,img)
    time.sleep(1)
