'''
@File: eval.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 4月 16, 2024
@HomePage: https://github.com/YanJieWen
'''

import json
import os
import os.path as osp
import sys

from tqdm import tqdm


import torch
import d2l.torch as d2l
import numpy as np


from create_datas import Coco_datasets
from model_utils.train_engine import install_detector
from model_utils.coco_eval import CocoEvaluator
from model_utils.train_utils import non_max_suppression,scale_coords

def summarize(self, catId=None):
    """
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    """

    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = self.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, :, catId, aind, mind]
            else:
                s = s[:, :, :, aind, mind]

        else:
            # dimension of recall: [TxKxAxM]
            s = self.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, catId, aind, mind]
            else:
                s = s[:, :, aind, mind]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])

        print_string = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
        return mean_s, print_string

    stats, print_list = [0] * 12, [""] * 12
    stats[0], print_list[0] = _summarize(1)
    stats[1], print_list[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
    stats[2], print_list[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
    stats[3], print_list[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
    stats[4], print_list[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[5], print_list[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
    stats[6], print_list[6] = _summarize(0, maxDets=self.params.maxDets[0])
    stats[7], print_list[7] = _summarize(0, maxDets=self.params.maxDets[1])
    stats[8], print_list[8] = _summarize(0, maxDets=self.params.maxDets[2])
    stats[9], print_list[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
    stats[10], print_list[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[11], print_list[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])

    print_info = "\n".join(print_list)

    if not self.eval:
        raise Exception('Please run accumulate() first')

    return stats, print_info


def main():
    device = d2l.try_gpu()
    print('Using %s device training' % device.type)

    label_json_path = './datasets/poc_classes.json'
    data_root = './datasets/POCOCO/'
    data_type = 'test'
    img_size = 640
    model = 'CSPDarknet'
    fpn = 'BackboneWithFPN'
    pt_path = './weights/ablation_dyhead/wo_channel/crashDet-23.pt'
    pred_path = osp.join('./analysis',f'pred_{model}.npy')

    #step1: 加载数据
    with open(label_json_path,'r') as r:
        clss_dict = json.load(r)
    cat_index = {v:k for k,v in clss_dict.items()}
    nc = len(cat_index)
    test_datasets = Coco_datasets.COCOdatasets(data_root,data_type,batch_size=1,img_size=img_size,rect=True)
    dataloader = torch.utils.data.DataLoader(test_datasets,batch_size=1,shuffle=False,pin_memory=True,
                                             collate_fn=test_datasets.collate_fn_yolo)
    coco = test_datasets.coco
    #step2:加载模型
    model = install_detector(backbone_name=model,fpn_name=fpn,nc=nc)
    assert osp.exists(pt_path), '%s is not exist, check the pt path' % pt_path
    weights_dict =torch.load(pt_path,map_location='cpu')
    weights_dict = weights_dict['model'] if 'model' in weights_dict.keys() else weights_dict
    model.load_state_dict(weights_dict,strict=False)
    model.to(device)

    #step3: 定义评估器
    coco_evaluator = CocoEvaluator(coco,['bbox'])
    cpu_device = torch.device("cpu")
    predictions = []
    #step4:开始评估
    model.eval()
    with torch.no_grad():
        for imgs,targets,paths,shapes,img_index in tqdm(dataloader,desc='validation...'):
            imgs = imgs.to(device).float()/255.0
            pred,_ = model(imgs)
            pred = non_max_suppression(pred, conf_thres=0.01, iou_thres=0.6, multi_label=False)

            outputs = []
            for index,p in enumerate(pred):#遍历每个样本
                if p is None:
                    p = torch.empty((0,6),device=cpu_device)
                    boxes = torch.empty((0,4),device=cpu_device)
                else:
                    boxes = p[:, :4]
                    # shapes: (h0, w0), ((h / h0, w / w0), pad)
                    boxes = scale_coords(imgs[index].shape[1:], boxes, shapes[index][0]).round()
                info = {"boxes": boxes.to(cpu_device),
                        "labels": p[:, 5].to(device=cpu_device, dtype=torch.int64)+1,
                        "scores": p[:, 4].to(cpu_device)}
                outputs.append(info)
            res = {img_id: output for img_id, output in zip(img_index, outputs)}#以dict{img_id: dict保存}
            predictions.append(res)
            coco_evaluator.update(res)
    #存储预测值便于后期分析
    preds = [np.zeros((0,7),dtype=np.float32)]*len(predictions)
    if not osp.isfile(pred_path):
        for i,meta_dict in enumerate(predictions):
            for img_id,info in meta_dict.items():
                img_id = torch.as_tensor([img_id]*info['boxes'].shape[0]).view(-1,1)
                preds[i] = torch.cat((img_id,info['boxes'],info['labels'].view(-1,1),info['scores'].view(-1,1)),dim=1).numpy()
        preds = np.concatenate(preds,axis=0)
        print("Saving predictions to %s for faster future loading as form [boxes,labels,scores]" % pred_path)
        np.save(pred_path,preds)
    else:
        raise ValueError(f'The predictions path has been saved at {pred_path},do not run same time!')



    #step5: 评估器统计精度
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    coco_eval = coco_evaluator.coco_eval["bbox"]
    # calculate COCO info for all classes
    coco_stats, print_coco = summarize(coco_eval)
    # calculate voc info for every classes(IoU=0.5)
    voc_map_info_list_ap5 = []
    voc_map_info_list_ap95 = []
    for i in range(len(cat_index)):
        stats, _ = summarize(coco_eval, catId=i)
        voc_map_info_list_ap5.append(" {:15}: {}".format(cat_index[i + 1], stats[1]))
        voc_map_info_list_ap95.append(" {:15}: {}".format(cat_index[i + 1], stats[0]))
    print_voc_5 = "\n".join(voc_map_info_list_ap5)
    print_voc_95 = '\n'.join(voc_map_info_list_ap95)
    print(print_voc_5)
    with open("record_mAP.txt", "w") as f:
        record_lines = ["COCO results:",
                        print_coco,
                        "",
                        "mAP(IoU=0.5) for each category:",
                        print_voc_5,
                        '',
                        "mAP(IoU=0.5:0.95) for each category:",
                        print_voc_95]
        f.write("\n".join(record_lines))

if __name__ == '__main__':
    main()