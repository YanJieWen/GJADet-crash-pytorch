'''
@File: Coco_datasets.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 3月 20, 2024
@HomePage: https://github.com/YanJieWen
'''
import copy
import json

import cv2
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import os
import os.path as osp
import numpy as np
from tqdm import tqdm

from model_utils.augers import Data_augmentation
from model_utils.data_tools import *


from PIL import Image,ImageDraw
import os.path as osp
import matplotlib.pyplot as plt



class COCOdatasets(Dataset):
    def __init__(self,root,dataset='train',batch_size=16,img_size=416,model_type='yolo',augment=False,rect=False,pad=0.0,**kwargs):
        '''
       支持coco以xyxy格式保存的voc并输出fasterrcnn格式；支持coco以xywh格式保存的coco并输出yolo格式xywh_rel
        :param root: './datasets/POCOCO/'
        :param dataset: train or val
        :param batch_size: 16
        :param img_size:512
        :param model_type: yolo
        :param augment: True
        :param rect:False
        :param pad:0.
        '''
        super(COCOdatasets,self).__init__(**kwargs)
        # assert dataset in ['train','val','test'], 'datasets must be train or val'
        anno_file = f'annotations/{dataset}.json'
        # anno_file = f'annotations/instances_{dataset}.json'
        self.anno_path = osp.join(root,anno_file)
        assert osp.isfile(self.anno_path),f'{anno_file} is not found'
        self.img_root = osp.join(root,f'{dataset}')
        assert osp.exists(self.img_root),f'{self.img_root} is not exists'

        self.mode = dataset
        self.img_size = img_size
        self.coco = COCO(self.anno_path)
        self.model_type = model_type #[yolo:[xr,yr,wr,hr],faster_rcnn:[xmin,ymin,xmax,ymax]]
        assert self.model_type in ['yolo','faster_rcnn'],'model type only supported yolo or faster_rcnn'
        #定义数据增强
        self.augment = augment
        self.rect = rect
        self.mosaic = False
        if self.augment and not self.rect:
            self.mosaic = True
            self.tss = [Data_augmentation() for _ in range(4)]
        elif self.augment and self.rect:
            self.tss = Data_augmentation()
        else:
            pass
        #映射coco的cat_id
        # idx_map = dict([kv[0],idx+1] for idx,kv in enumerate(self.coco.cats.items()))
        # new_cls_dict = {}
        # for id,(k,v) in enumerate(self.coco.cats.items()):
        #     new_cls_dict[id+1] = {'id':idx_map.get(v['id']),'name':v['name'],'supercategory':v['supercategory']}
        # self.coco.cats = new_cls_dict
        # with open('coco_classes.json','w') as w:
        #     json.dump({k:v['name'] for k,v in new_cls_dict.items()},w)
        # w.close()
        # print(new_cls_dict)

        #更新anns
        # for k,v in self.coco.anns.items():
        #     v['category_id'] = idx_map.get(v['category_id'])

        data_classes = dict([v['id'],v['name']] for k,v in self.coco.cats.items())
        max_index = max(data_classes.keys())
        coco_classes = {}
        for k in range(1,max_index + 1):
            if k in data_classes:
                coco_classes[k] = data_classes[k]
            else:
                coco_classes[k] = 'N/A'
        self.coco_classes = coco_classes
        ids = sorted(self.coco.imgs.keys())
        #移除面积非常小的id
        if dataset=='train':
            valid_ids = coco_remove_images_without_annotations(self.coco,ids)
            self.ids  = valid_ids
        else:
            self.ids = ids

        #对图像的形状进行设置
        n = len(self.ids)
        bi = np.floor(np.arange(n)/batch_size).astype(int)
        nb = bi[-1]+1
        self.n = n
        self.batch = bi
        sp = osp.join(root,f'poc_{dataset}_data.shapes')
        try:
            with open(sp,'r') as r:
                s = [_s.split() for _s in r.read().splitlines()]
                assert len(s)==n, 'the shape file is not matched images'
        except Exception as e:
            img_ids = tqdm(self.ids,desc='caching the image shapes')
            s = [cv2.imread(osp.join(self.img_root,self.coco.loadImgs(id)[0]['file_name'])).transpose(1,0,2) .shape[:2] for id in img_ids]
            np.savetxt(sp,s,fmt='%g')
        self.shapes = np.array(s,dtype=np.float64)
        #如果实现rect training
        if self.rect:
            s = self.shapes
            ar = s[:,1]/s[:,0]
            irect = ar.argsort()
            self.ids = [self.ids[i] for i in irect]#少批量运行
            self.shapes = s[irect]
            ar = ar[irect]

            shapes = [[1,1]]*nb
            for i in range(nb):
                ari = ar[bi==i]
                mini,maxi = ari.min(),ari.max()
                if maxi<1:
                    shapes[i] = [maxi,1]
                elif mini>1:
                    shapes[i] = [1,1/mini]
            self.batch_shapes = np.ceil(np.array(shapes)*img_size/32.+pad).astype(int)*32
    def parse_target(self,
                     img_id:int,
                     coco_targets:list,
                     w:int,
                     h:int):
        assert w>0 and h>0, 'w/h can not meet required'
        boxes = [obj['bbox'] for obj in coco_targets]
        boxes = torch.as_tensor(boxes,dtype=torch.float32).reshape(-1,4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)#xmin,ymin,xmax,ymax for faster-rcnn
        if self.model_type=='yolo': #relx,rely,relw,relh
            xc_rel = (boxes[:,0:1]+(boxes[:,2:3]-boxes[:,0:1])/2)/w
            yc_rel = (boxes[:,1:2]+(boxes[:,3:]-boxes[:,1:2])/2)/h
            w_rel = (boxes[:,2:3]-boxes[:,0:1])/w
            h_rel = (boxes[:,3:]-boxes[:,1:2])/h
            boxes = torch.concatenate((xc_rel,yc_rel,w_rel,h_rel),dim=1)
        elif self.model_type=='faster_rcnn':
            boxes[:,0::2]/=w #全部转为相对坐标
            boxes[:,1::2]/=h
            boxes = torch.as_tensor(boxes)
        else:
            raise ValueError('model type must be faster_rcnn or yolo')
        classes = [obj['category_id'] for obj in coco_targets]
        classes = torch.as_tensor(classes,dtype=torch.int64)
        area = torch.tensor([obj['area'] for obj in coco_targets],dtype=torch.float32)
        iscrowd = torch.tensor([obj['iscrowd'] for obj in coco_targets],dtype=torch.int64)
        segmentation = [obj['segmentation'] for obj in coco_targets if len(obj['segmentation'])!=0]
        if len(segmentation)!=0:
            masks = convert_coco_poly_mask(segmentation,h,w)#多边形转为mask蒙版
        else:
            masks =  None
        target = {}
        target['boxes'] = boxes
        target['classes'] = classes
        target['masks'] = masks if masks is not None else []
        target['image_id'] = torch.tensor([img_id])
        target['area'] = area
        target['iscrowd'] = iscrowd
        return target

    def __len__(self):
        return len(self.ids)


    def __getitem__(self, index):
        if self.mosaic:
            img,labels = load_mosaic(self,index,self.tss)
            if 'image_id' not in labels.keys():
                labels['image_id'] = self.ids[index]
            shapes = None
        else:
            img,(h0,w0),(h,w) = load_image(self,index)
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size
            img, ratio, pad = letterbox(img, shape, auto=False, scale_up=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)
            img_id = self.ids[index]
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            coco_targets = self.coco.loadAnns(ann_ids)
            x = self.parse_target(img_id,coco_targets,w0,h0)
            _labels = copy.deepcopy(x['boxes'])
            if x['boxes'].size()[0]>0:
                if self.model_type=='yolo':
                    _labels[:, 0] = ratio[0] * w * (x['boxes'][:, 0] - x['boxes'][:, 2] / 2) + pad[0]
                    _labels[:, 1] = ratio[1] * h * (x['boxes'][:, 1] - x['boxes'][:, 3] / 2) + pad[1]
                    _labels[:, 2] = ratio[0] * w * (x['boxes'][:, 0] + x['boxes'][:, 2] / 2) + pad[0]
                    _labels[:, 3] = ratio[1] * h * (x['boxes'][:, 1] + x['boxes'][:, 3] / 2) + pad[1]
                else:
                    _labels[:, 0::2] = ratio[0] * w * x['boxes'][:,0::2]+pad[0]
                    _labels[:,1::2] = ratio[1]*h*x['boxes'][:,1::2]+pad[1]
            labels = copy.deepcopy(x)
            labels['boxes'] = _labels
        if self.augment:
            if not self.mosaic:
                _dw, _dh = np.ceil(pad[0] / 2), np.ceil(pad[1] / 2)
                se_w = (int(_dw), int(-_dw)) if _dw != 0 else (0, -1)
                se_h = (int(_dh), int(-_dh)) if _dh != 0 else (0, -1)
                img[se_h[0]:se_h[1], se_w[0]:se_w[1]], labels['boxes'] = self.tss(img[se_h[0]:se_h[1], se_w[0]:se_w[1]], labels['boxes'].numpy())
                # 添加限制条件防止越界
                np.clip(labels['boxes'][:, [0, 2]],0,img.shape[1],out=labels['boxes'][:, [0, 2]])#缩放到指定大小，范围不再是0因为进行了上下填充
                np.clip(labels['boxes'][:, [1, 3]],0,img.shape[0],out=labels['boxes'][:, [1, 3]])

                labels['boxes'] = torch.as_tensor(labels['boxes'])
        if self.model_type=='yolo':#转为yolo格式保存
            nl = len(labels['boxes'])
            if nl:
                labels['boxes'] = xyxy2xywh(labels['boxes'])
                labels['boxes'][:,[1,3]]/=img.shape[0]
                labels['boxes'][:,[0,2]]/=img.shape[1]
            labels_out =torch.zeros((nl,6))
            if nl:
                x = torch.concatenate((labels['classes'].reshape(-1,1),labels['boxes']),dim=-1)
                labels_out[:,1:] = x
            img = img[:,:,::-1].transpose(2,0,1)
            img = np.ascontiguousarray(img)
            return torch.from_numpy(img),labels_out,self.ids[index],shapes,self.ids[index]

        elif self.model_type=='faster_rcnn':#转为faster_rcnn所需要的输出格式
            box_info  = labels['boxes']
            area = (box_info[:, 3] - box_info[:, 1]) * (box_info[:, 2] - box_info[:, 0])
            labels['area'] = area
            return img,labels

    @staticmethod
    def collate_fn_rcnn(batch):#fasterrcnn将所有图像和target制作成列表格式
        return tuple(zip(*batch))
    @staticmethod
    def collate_fn_yolo(batch): #yolo将所有的label混合在一起使用
        img,label,path,shapes,index = zip(*batch)
        for i,l in enumerate(label):
            l[:,0] = i #给图一个顺序索引
        return torch.stack(img,0),torch.cat(label,0),path,shapes,index



# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     import PIL.ImageDraw as Imagedraw
#     from PIL import ImageColor,ImageFont
#     from torchvision import transforms

#     STANDARD_COLORS = [
#                 'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
#                 'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
#                 'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
#                 'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
#                 'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
#                 'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
#                 'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
#                 'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
#                 'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
#                 'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
#                 'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
#                 'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
#                 'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
#                 'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
#                 'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
#                 'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
#                 'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
#                 'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
#                 'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
#                 'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
#                 'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
#                 'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
#                 'WhiteSmoke', 'Yellow', 'YellowGreen'
#             ]
#     root = '../datasets/POCOCO/'
#     # dataset = 'val'
#     dataset = 'val'
#     batch_size = 8
#     img_size = 512
#     model_type = 'yolo'
#     rect = False
#     data = COCOdatasets(root,dataset,batch_size,img_size,model_type,rect=rect, augment=True)
#     coco = data.coco
#     # ids = list(sorted(coco.imgs.keys()))
#     # print(data.coco.cats)
#     dataloader = torch.utils.data.DataLoader(data,batch_size=batch_size,num_workers=0,shuffle=rect,pin_memory=True,collate_fn=data.collate_fn_yolo)
#     samples = next(iter(dataloader))
#     # print(data.batch_shapes[0])
#     for im_id in range(batch_size):
#         # im_id = 5
#         keep = samples[1][:,0]==im_id
#         img = samples[0][im_id]
#         gt = samples[1][keep]
#         # print(gt)
#         # print(samples[2][im_id],samples[3][im_id],samples[4][im_id])
#         h, w = img.shape[1:]
#         coco_classes = dict([(v['id'],v['name']) for k,v in data.coco.cats.items()])
#         # print(gt)
#         colors = [ImageColor.getrgb(STANDARD_COLORS[cls % len(STANDARD_COLORS)]) for cls in
#                   np.asarray(gt[:, 1].numpy(), dtype=int)]
#         img = transforms.ToPILImage()(img)
#         draw = Imagedraw.Draw(img)
#         scores = torch.ones(gt.shape[0])
#         font = ImageFont.truetype('arial.ttf', 12)
#         # print(gt[:,1])
#         for box, cls, color, score in zip(gt[:, 2:], gt[:, 1], colors, scores):
#             norm_cx, norm_cy, norm_w, norm_h = box
#             left, top = (norm_cx - norm_w / 2) * w, (norm_cy - norm_h / 2) * h
#             right, bottom = (norm_cx + norm_w / 2) * w, (norm_cy + norm_h / 2) * h
#             draw.line([(left, top), (left, bottom), (right, bottom),
#                                    (right, top), (left, top)],width=1,fill=color)
#             draw.text((left,top), coco_classes.get(int(cls)), color,font=font)
#         plt.axis('off')
#         plt.imshow(img)
#         plt.show()

#
#     ann_json = '../datasets/POCOCO/annotations/test.json'
#     img_root = '../datasets/POCOCO/test'
#     coco = COCO(ann_json)
#     ids = list(sorted(coco.imgs.keys()))
#     coco_classes = dict([(v['id'],v['name']) for k,v in coco.cats.items()])
#     for i in ids[:3]:
#         ann_ids = coco.getAnnIds(i)
#         tgts = coco.loadAnns(ann_ids)
#         path = coco.loadImgs(i)[0]['file_name']
#         img_path =  osp.join(img_root,path)
#         _img = Image.open(img_path).convert('RGB')
#         draw = ImageDraw.Draw(_img)
#         for tgt in tgts:
#             x,y,w,h = tgt['bbox']
#             xmin,ymin,xmax,ymax = x,y,x+w,y+h
#             draw.rectangle((xmin,ymin,xmax,ymax),width=3)
#             draw.text((xmin,ymin),coco_classes.get(tgt['category_id']))
#         plt.imshow(_img)
#         plt.show()
