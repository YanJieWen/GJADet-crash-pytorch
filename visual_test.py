'''
@File: visual_test.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 3月 31, 2024
@HomePage: https://github.com/YanJieWen
'''


import matplotlib.pyplot as plt
import PIL.ImageDraw as Imagedraw
from PIL import ImageColor,ImageFont
from torchvision import transforms
import numpy as np

import json

STANDARD_COLORS = [
            'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
            'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
            'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
            'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
            'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
            'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
            'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
            'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
            'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
            'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
            'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
            'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
            'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
            'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
            'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
            'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
            'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
            'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
            'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
            'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
            'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
            'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
            'WhiteSmoke', 'Yellow', 'YellowGreen']

def plot_visuals(imgs,tgts):
    '''
    可视化损失无限大的图像
    :param imgs: [B,C,H,W]
    :param tgts: [n,6]
    :return:
    '''
    with open('./datasets/poc_classes.json','r') as r:
        class_dict = json.load(r)
    poc_classes = {v:k for k,v in class_dict.items()}
    for i,img in enumerate(imgs):
        h, w = img.shape[1:]
        tgt = tgts[tgts[:,0]==i]
        img = img.squeeze(dim=0)
        img = transforms.ToPILImage()(img)
        colors = [ImageColor.getrgb(STANDARD_COLORS[cls % len(STANDARD_COLORS)]) for cls in
                      np.asarray(tgt[:, 1].detach().to('cpu').numpy(), dtype=int)]
        draw = Imagedraw.Draw(img)
        font = ImageFont.truetype('arial.ttf', 3)
        for box, cls, color,  in zip(tgt[:, 2:], tgt[:, 1], colors):
            norm_cx, norm_cy, norm_w, norm_h = box
            left, top = (norm_cx - norm_w / 2) * w, (norm_cy - norm_h / 2) * h
            right, bottom = (norm_cx + norm_w / 2) * w, (norm_cy + norm_h / 2) * h
            draw.line([(left, top), (left, bottom), (right, bottom),
                                           (right, top), (left, top)],width=1,fill=color)
            draw.text((left, top), poc_classes.get(int(cls)), color, font=font)
        plt.imshow(img)
        plt.show()