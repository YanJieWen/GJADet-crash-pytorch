'''
@File: kmeans_coco.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 4月 10, 2024
@HomePage: https://github.com/YanJieWen
'''
import numpy as np

from create_datas.Coco_datasets import COCOdatasets
from tqdm import tqdm
import os.path as osp
import os
import random

def get_info(dataset):
    im_wh_list = []
    boxes_wh_list = []
    pbar = tqdm(range(len(dataset)),desc='read data info')
    for idx in pbar:
        wh = []
        img,labels,_,_,_ = dataset[idx]
        _,h,w = img.shape
        for l in labels:
            wh.append([l[4].item(),l[5].item()])
        if len(wh)==0:
            continue
        im_wh_list.append([w,h])
        boxes_wh_list.append(wh)
    return im_wh_list,boxes_wh_list

def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = np.minimum(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)

def k_means_iou(boxes,k,dist=np.median,max_iter=500):
    box_number = boxes.shape[0]
    last_nearst = np.zeros((box_number,))
    clusters = boxes[np.random.choice(box_number,k,replace=False)]
    _iter = 0
    while True and _iter<max_iter:
        distance = 1-wh_iou(boxes,clusters)
        current_nearst = np.argmin(distance,axis=1)
        if (last_nearst==current_nearst).all() or _iter<max_iter:
            break
        for cluster in range(k):
            clusters[cluster] = dist(boxes[current_nearst==cluster],axis=0)
        last_nearst = current_nearst
        _iter+=1
    return clusters

def anchor_fitness(k: np.ndarray, wh: np.ndarray, thr: float):  # mutation fitness
    r = wh[:, None] / k[None]
    x = np.minimum(r, 1. / r).min(2)  # ratio metric
    # x = wh_iou(wh, k)  # iou metric
    best = x.max(1)
    f = (best * (best > thr).astype(np.float32)).mean()  # fitness
    bpr = (best > thr).astype(np.float32).mean()  # best possible recall
    return f, bpr

if __name__ == '__main__':

    #超参数
    img_size = 640
    ana_root = './analysis/'
    if not osp.exists(ana_root):
        os.makedirs(ana_root)
    dataset_type = 'val'
    data_root = './datasets/POCOCO/'
    kc = 9
    thr = 0.25
    gen = 1000

    dataset = COCOdatasets(root=data_root,dataset=dataset_type,img_size=img_size,model_type='yolo',
                           rect=True,augment=True)
    img_shape_file = osp.join(ana_root,f'imgsp_{dataset_type}.shapes')
    label_file = osp.join(ana_root,f'lb_{dataset_type}.labels')
    try:
        with open(img_shape_file,'r') as r:
            im_wh_list = [_s.split() for _s in r.read().splitlines()]
            r.close()
        with open(label_file,'r') as r:
            boxes_wh_list = [_s.split() for _s in r.read().splitlines()]
            r.close()
    except Exception as e:
        im_wh_list, boxes_wh_list = get_info(dataset)
        np.savetxt(img_shape_file,im_wh_list,fmt='%g')
        im_wh = np.array(im_wh_list, dtype=np.float32)
        boxes_wh_list = np.concatenate([l * s for s, l in zip(im_wh, boxes_wh_list)])
        np.savetxt(label_file,boxes_wh_list,fmt='%g')
    if isinstance(boxes_wh_list,list):
        wh0 = np.array(boxes_wh_list,dtype=np.float32)
    else:
        wh0 = boxes_wh_list

    k_file = osp.join(ana_root,f'kmeans_{dataset_type}.cls')
    i = (wh0 < 3.0).any(1).sum()
    if i:
        print(f'WARNING: Extremely small objects found. {i} of {len(wh0)} labels are < 3 pixels in size.')
    wh = wh0[(wh0 >= 2.0).any(1)]
    try:
        with open(k_file,'r') as r:
            k = [_s.split() for _s in r.read().splitlines()]
            r.close()
        k = np.array(k,dtype=np.float32)
    except Exception as e:
        k = k_means_iou(wh, k=kc)
        k = k[np.argsort(k.prod(1))]
        np.savetxt(k_file,k,fmt='%g')

    f, bpr = anchor_fitness(k, wh, thr)  # 计算平均适应度和最优召回率
    print("kmeans: " + " ".join([f"[{int(i[0])}, {int(i[1])}]" for i in k]))
    print(f"fitness: {f:.5f}, best possible recall: {bpr:.5f}")

    # Evolve
    # 遗传算法(在kmeans的结果基础上变异mutation)
    npr = np.random
    f, sh, mp, s = anchor_fitness(k, wh, thr)[0], k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    pbar = tqdm(range(gen), desc=f'Evolving anchors with Genetic Algorithm:')  # progress bar
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(sh) < mp) * random.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg, bpr = anchor_fitness(kg, wh, thr)
        if fg > f:
            f, k = fg, kg.copy()
            pbar.desc = f'Evolving anchors with Genetic Algorithm: fitness = {f:.4f}'

    # 按面积排序
    k = k[np.argsort(k.prod(1))]  # sort small to large
    print("genetic: " + " ".join([f"[{int(i[0])}, {int(i[1])}]" for i in k]))
    print(f"fitness: {f:.5f}, best possible recall: {bpr:.5f}")





