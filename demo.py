'''
@File: demo.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 4月 04, 2024
@HomePage: https://github.com/YanJieWen
'''


import os
import json
import sys
import time

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from visualization.grad_cam import *

import torchvision.transforms as ts


from model_utils.data_tools import letterbox
from model_utils.train_utils import non_max_suppression,scale_coords
from model_utils.train_engine import *
from model_utils.visual_utils import draw_objs


def main():
    img_size = 640
    weight_path = './weights/cspdark_dyhead_our/crashDet-35.pt'
    json_path = './datasets/poc_classes.json'
    img_path = './datasets/POCOCO/test/002277.jpg'
    # img_path = './datasets/00000672.jpg'
    output_dir = './demo/results/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(json_path,'r') as f:
        class_dict = json.load(f)
    category_index = {str(v): str(k) for k, v in class_dict.items()}#{1:str}
    input_size = (img_size,img_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = install_detector(backbone_name='CSPDarknet',fpn_name='BackboneWithFPN',verbose=False,nc=len(class_dict))
    weights_dict = torch.load(weight_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict, strict=False)
    model.to(device)
    model.eval()
    with torch.no_grad():
        img = torch.zeros((1,3,img_size,img_size),device=device)
        model(img)#预热
        img_o = cv2.imread(img_path)
        img = letterbox(img_o,new_shape=input_size,auto=True,color=(0,0,0))[0]
        visual_img = img.copy()
        img = img[:,:,::-1].transpose(2,0,1)
        img =np.ascontiguousarray(img)
        img = torch.as_tensor(img).to(device).float()
        img /= 255.
        img = img.unsqueeze(0)
        #===============grad_cam===============
        target_layers = [model.model[9]]#12,16,20,21->4,6,9
        cam = GradCAM(model=model, target_layers=target_layers,use_cuda=True)
        grayscale_cam = cam(input_tensor=img)
        grayscale_cam = grayscale_cam[0, :]
        _original_img_visual = np.array(visual_img, dtype=np.uint8).copy()
        visualization = show_cam_on_image(_original_img_visual.astype(dtype=np.float32) / 255.,
                                          grayscale_cam,
                                          use_rgb=True)
        plt.axis('off')
        plt.imshow(visualization)
        save_fig('002')
        plt.show()
        #=============================================
        pred, _ = model(img)
        pred = non_max_suppression(pred,conf_thres=0.1,iou_thres=0.4,multi_label=True)[0]

        if pred is None:
            print('no tgt detected')
            sys.exit(0)
        #缩放边界框
        pred[:,:4] = scale_coords(img.shape[2:],pred[:,:4],img_o.shape).round()
        bboxes = pred[:, :4].detach().cpu().numpy()
        scores = pred[:, 4].detach().cpu().numpy()
        classes = pred[:, 5].detach().cpu().numpy().astype(np.int) + 1
        print(classes)
        pil_img = ts.ToPILImage()(img_o[:,:,::-1])
        #可视化demo结果
        plot_img = draw_objs(pil_img,bboxes,classes,scores,category_index=category_index,box_thresh=0.3,
                             line_thickness=3,font='arial.ttf',font_size=10)
        plt.imshow(plot_img)
        plt.show()
        plot_img.save(f'{output_dir}{os.path.basename(img_path)}')
if __name__ == '__main__':
    main()
