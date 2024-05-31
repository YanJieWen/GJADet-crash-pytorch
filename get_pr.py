#!/usr/bin/env python
# coding: utf-8
'''
@Time: 2024/5/17 21:38
@File: get_pr.py
@Author: Yanjie Wen
@Software: PyCharm
@Homepage: https://github.com/YanJieWen
@Paperpage: https://scholar.google.com.hk/citations?user=RM4K94oAAAAJ&hl=zh-CN
@E-mail: obitowen@csu.edu.cn
'''

from create_datas import Coco_datasets

import os
import numpy as np
from pycocotools.cocoeval import COCOeval
import pandas as pd
from glob import glob


def pr_out(data_root,data_type,thr=0.75,save_root='./weights/cspdark_dyhead_our/'):
    '''
    需要提前将预测结果保存为json文件
    precisions[T, R, K, A, M]
    T: iou thresholds [0.5 : 0.05 : 0.95], idx from 0 to 9
    R: recall thresholds [0 : 0.01 : 1], idx from 0 to 100
    K: category, idx from 0 to ...
    A: area range, (all, small, medium, large), idx from 0 to 3
    M: max dets, (1, 10, 100), idx from 0 to 2
    :param data_root: './datasets/POCOCO'
    :param data_type:'test'
    :param thr:[0.5:0.05:0.95],-1 当为-1的时候为为0.5:0.95的评估
    :param save_root:'./weights/cspdark_dyhead_our/'
    :return:
    '''
    # with open(json_path,'r') as r:
    #     data = json.load(r)
    # r.close()
    test_datasets = Coco_datasets.COCOdatasets(data_root, data_type, batch_size=1,
                                               img_size=512, rect=True)
    coco = test_datasets.coco
    json_path = save_root
    coco_dt = coco.loadRes(json_path)
    coco_eval = COCOeval(coco, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    p_value = coco_eval.eval['precision']
    # r_value = coco_eval.eval['recall']
    recall = np.mat(np.arange(0.0, 1.01, 0.01)).T
    # assert thr*100%5==0 and 0.5<=thr<=0.95,f'The f{thr} is not matched'
    M = 2
    if thr==-1:
        map_all_pr = np.mean(p_value[:, :, :, 0, M], axis=0)
    else:
        T = int((thr - 0.5) / 0.05)

        map_all_pr = p_value[T, :, :, 0, M]
    data = np.hstack((np.hstack((recall, map_all_pr)),
                      np.mat(np.mean(map_all_pr, axis=1)).T))#取所有类的ap以及平均ap曲线
    df = pd.DataFrame(data)
    save_path = os.path.join('./weights/baselines/ap5to95','pr_curve.xlsx')
    df.to_excel(save_path, index=False)


def main():
    data_root = './datasets/POCOCO'
    data_type = 'test'
    thr = -1
    save_root = './weights/baselines/jsons/yolov3.json'
    pr_out(data_root,data_type,thr,save_root)

if __name__ == '__main__':
    main()



