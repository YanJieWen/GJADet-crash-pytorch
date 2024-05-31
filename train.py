'''
@File: main.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 4月 3, 2024
@HomePage: https://github.com/YanJieWen
'''

import glob
import os
import os.path as osp
import math
import json

import torch

from collections import OrderedDict
import d2l.torch as d2l
from torch.utils.tensorboard import SummaryWriter

from model_utils.parse_config import dynamica_install
from tools import *
from model_utils.parse_config import *
from create_models.Yolov3_spp import YOLOLayer
from model_utils.train_engine import *

#定义需要保存的日志文件
log_path = './log.txt'
log = open(log_path,'a+')



#加载损失函数用于训练(损失函数，边界框采样,迁移学习)
def train(opt):
	device = d2l.try_gpu()
	print(f'Using {device.type} device training')
	#step1: 定义参数化文件
	wdir = './weights'+os.sep
	if not osp.exists(wdir):
		os.makedirs(wdir)
	best = wdir+'best.pt'
	if not osp.isfile(best):
		print_log('The model has not been achive best...',log)
	else:
		raise ValueError(f'The model has been achieved best...')
	results_file = 'results{}.txt'.format(get_timestring())

	first_epochs = opt.first_epochs
	second_epochs = opt.second_epochs
	batch_size = opt.batch_size
	accumulate = max(round(64/batch_size),1)# 经过n次累计更新网络参数
	weights = opt.weights #初始化训练权重
	imgsz_train = opt.img_size
	imgsz_test = opt.img_size
	multi_scale = opt.multi_scale#是否启用多尺度训练
	data_root = opt.data_root
	model_type = opt.model_type
	augment = opt.augment
	rect = opt.rect
	nc = opt.nc

	gs = 32
	assert math.fmod(imgsz_test,gs)==0,'--img_size %g must be  a %g -multiple' % (imgsz_test,gs) #计算余数
	grid_min,grid_max = imgsz_test//gs,imgsz_test//gs
	if multi_scale:#如果启用动态尺度训练
		imgz_min = opt.img_size//1.5
		imgz_max = opt.img_size//0.667
		grid_min,grid_max = imgz_min//gs,imgz_max//gs
		imgsz_min,imgsz_max = int(grid_min*gs),int(grid_max*gs)
		imgsz_train = imgsz_max#最大尺寸定为初始尺寸
		print("Using multi_scale training, image range[{}, {}]".format(imgsz_min, imgsz_max))
	else:
		print('The ratio single scale training range[%g,%g]' % (grid_min,grid_max))
	# class_path = opt.class_path
	# with open(class_path,'r') as r:
	# 	data = json.load(r)
	# class_dict = {v:k for k,v in data.items()}#dict[str(int):cls]
	for f in glob.glob(results_file):
		os.remove(f)

	#step2: 模型定义以及权重冻结
	model = install_detector(backbone_name=opt.backbone_name,fpn_name=opt.fpn_name,nc=opt.nc).to(device)
	print_log(f'{"*" * 15} {opt.backbone_name} with {opt.fpn_name} has been installed {"*" * 15}', log)
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	#  first frozen backbone and train 12 epochs                   #
	#  首先冻结前置特征提取网络权重（backbone），训练FPN网络 #
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	if opt.freeze_layers and opt.backbone_name not in ('Darknet','SwinTransformer'): #是否冻结权重
		for param in model.backbonewithfpn.body.parameters():
			param.require_grad = False
	elif opt.freeze_layers and opt.backbone_name=='SwinTransformer':
		for param in model.backbone.parameters():
			param.require_grad = False
	elif opt.freeze_layers and opt.backbone_name=='Darknet': #不分两阶段训练,即在第二阶段darknet不做任何变化
		output_layer_indices = [idx - 1 for idx, module in enumerate(model.module_list) if
								isinstance(module, YOLOLayer)]
		freeze_layer_indeces = [x for x in range(len(model.module_list)) if
								(x not in output_layer_indices) and
								(x - 1 not in output_layer_indices)]
		for idx in freeze_layer_indeces:#冻结除predictor和YOLOlayer之外的所有层
			for parameter in model.module_list[idx].parameters():
				parameter.requires_grad_(False)
	else:#从头开始训练
		pass
		# 如果freeze_layer为False，默认仅训练除darknet53之后的部分
		# 若要训练全部权重，删除以下代码
		# darknet_end_layer = 74  # only yolov3spp cfg
		# # Freeze darknet53 layers
		# # 总共训练21x3+3x2=69个parameters
		# for idx in range(darknet_end_layer + 1):  # [0, 74]
		# 	for parameter in model.module_list[idx].parameters():
		# 		parameter.requires_grad_(False)

	#step3:定义学习器
	pg = [p for p in model.parameters() if p.requires_grad]
	optimizer = torch.optim.SGD(pg, lr=0.001, momentum=0.937,
						  weight_decay=0.0005, nesterov=True)
	# optimizer = torch.optim.Adagrad(pg,lr=0.001,weight_decay=0.0005)
	scaler = torch.cuda.amp.GradScaler() if opt.amp else None
	#定义数据集-->以coco数据格式为例，动态加载数据集
	train_data_dict = {'root':data_root,'dataset':'train','batch_size':batch_size,'img_size':imgsz_train,'model_type':model_type,
					   'augment':augment,'rect':rect}
	val_data_dict = {'root':data_root,'dataset':'val','batch_size':batch_size,'img_size':imgsz_test,'model_type':model_type,
					   'augment':False,'rect':True}
	train_loader,_ = install_dataloader('coco',train_data_dict,'yolo')
	val_loader,val_dataset = install_dataloader('coco',val_data_dict,'yolo')

	hyp = {'obj_weight': 64.3*imgsz_test/320,'cls_weight':37.4*nc/80,'loc_weight':3.54,'obj_name':opt.obj_loss,
		   'cls_name':opt.cls_loss,'loc_name':opt.loc_loss}
	model.nc = nc
	model.hp = hyp
	#定义目标损失函数和分类损失函数
	#==================================
	#focal loss参数
	# obj_loss_dict = {'gamma':0.0,'alpha':0.25,'reduction':'mean','loss_weight':1.0}#这里的weight是针对每个样本损失的weight而不是样本的整体损失weight
	# cls_loss_dict = {'gamma':0.0,'alpha':0.25,'reduction':'mean','loss_weight':1.0}
	# #quality -loss参数
	# obj_loss_dict = {'beta': 2.0, 'reduction': 'mean', 'loss_weight': 1.0}
	# # cls_loss_dict = {'beta': 2.0, 'reduction': 'mean', 'loss_weight': 1.0}
	# #eql-q参数
	# # eql参数
	obj_loss_dict = {'beta': 2.0, 'reduction': 'mean', 'loss_weight': 1.0}
	cls_loss_dict = {'num_classes': nc, 'gamma': 5,'mu':0.8,'alpha':4.0,'beta':2}
	loc_loss_dict = {'x1y1x2y2':False,'CIoU':True}#GIoU,DIoU,if no parse IOU
	obj_loss_fun = install_loss_computing(opt.obj_loss,obj_loss_dict)
	cls_loss_fun = install_loss_computing(opt.cls_loss,cls_loss_dict)
	loc_loss_fun = install_loss_computing(opt.loc_loss,loc_loss_dict)

	coco = val_dataset.coco#用于评估使用
	print('starting  training for one-stage epochs...' )
	#step:4开始训练
	for epoch in range(opt.start_epochs,first_epochs):
		mloss,lr = train_one_epoch(model,optimizer,train_loader,device,epoch,accumulate=accumulate,img_size=imgsz_train,
								   multi_scale=multi_scale,grid_min=grid_min,grid_max=grid_max,gs=gs,print_feq=50,
								   warmup=True,scaler=scaler,obj_func=obj_loss_fun,cls_func=cls_loss_fun,loc_func=loc_loss_fun) #4*tensor&items
		# scheduler.step()
	# #加载评估指标用于推理(NMS,Soft-Nms)
		if not opt.notest or epoch == first_epochs-1:
			result_info = evaluate(model,val_loader,coco=coco,device=device)
			coco_mAP = result_info[0]#0.5:0.95
			voc_mAP = result_info[1]
			coco_mAR = result_info[8]

			if tb_writer:
				tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss', 'train/loss', "learning_rate",
						"mAP@[IoU=0.50:0.95]", "mAP@[IoU=0.5]", "mAR@[IoU=0.50:0.95]"]
				for x, tag in zip(mloss.tolist() + [lr, coco_mAP, voc_mAP, coco_mAR], tags):
					tb_writer.add_scalar(tag, x, epoch)
				with open(results_file,'a') as f:
					result_info = [str(round(i, 4)) for i in result_info + [mloss.tolist()[-1]]] + [str(round(lr, 6))]
					txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
					f.write(txt + "\n")
			#对于第一轮训练保存每一个epoch的权重
			with open(results_file,'r') as f:
				save_files = {'model': model.state_dict(),
							'optimizer':optimizer.state_dict(),
							'training_results': f.read(),
							'epoch': epoch,
							'best_map': coco_mAP}
				if opt.amp:
					save_files["scaler"] = scaler.state_dict()
				torch.save(save_files,f'{wdir}crashDet-{int(epoch)}.pt')
			print_log(f'{"="*15}epoch:{int(epoch)}--mAP:{round(coco_mAP,4)}--the first stage{"="*15}',log)
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	#  解耦骨干网络: 解冻顶部的特征提取网络，训练整个网络权重,如果是Darknet则继续训练#
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	decouple_training(model,opt.backbone_name)
	pg = [parm for parm in model.parameters() if parm.requires_grad]
	optimizer = torch.optim.SGD(pg, lr=0.01, momentum=0.937,
								weight_decay=0.0005, nesterov=True)
	scaler = torch.cuda.amp.GradScaler() if opt.amp else None
	lf = lambda x: ((1 + math.cos(x * math.pi / (second_epochs-first_epochs))) / 2) * (1 - 0.01) + 0.01
	scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
	init_epochs,best_map = load_weights(model,optimizer,results_file,opt.second_epochs,
										opt.first_epochs,scaler,opt.amp,weights,device)
	scheduler.last_epoch = init_epochs
	opt.start_epochs = init_epochs
	print('starting  training for second-stage epochs...')
	for epoch in range(opt.start_epochs,second_epochs,1):
		mloss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, accumulate=accumulate,
									img_size=imgsz_train,
									multi_scale=multi_scale, grid_min=grid_min, grid_max=grid_max, gs=gs, print_feq=50,
									warmup=True, scaler=scaler, obj_func=obj_loss_fun, cls_func=cls_loss_fun,
									loc_func=loc_loss_fun)
		scheduler.step()
		if opt.notest is False or epoch == second_epochs - 1:
			result_info = evaluate(model, val_loader, coco=coco, device=device)
			coco_mAP = result_info[0]  # 0.5:0.95
			voc_mAP = result_info[1]
			coco_mAR = result_info[8]
			if tb_writer:
				tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss', 'train/loss', "learning_rate",
						"mAP@[IoU=0.50:0.95]", "mAP@[IoU=0.5]", "mAR@[IoU=0.50:0.95]"]
				for x, tag in zip(mloss.tolist() + [lr, coco_mAP, voc_mAP, coco_mAR], tags):
					tb_writer.add_scalar(tag, x, epoch)
				with open(results_file,'a') as f:
					result_info = [str(round(i, 4)) for i in result_info + [mloss.tolist()[-1]]] + [str(round(lr, 6))]
					txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
					f.write(txt + "\n")
				if coco_mAP>best_map:
					best_map = coco_mAP
				if opt.save_best is False:
					with open(results_file, 'r') as f:
						save_files = {
							'model': model.state_dict(),
							'optimizer': optimizer.state_dict(),
							'training_results': f.read(),
							'epoch': epoch,
							'best_map': best_map}
						if opt.amp:
							save_files["scaler"] = scaler.state_dict()
						torch.save(save_files, f"{wdir}crashDet-{int(epoch)}.pt")
				else:
					if best_map == coco_mAP:
						with open(results_file, 'r') as f:
							save_files = {
								'model': model.state_dict(),
								'optimizer': optimizer.state_dict(),
								'training_results': f.read(),
								'epoch': epoch,
								'best_map': best_map}
							if opt.amp:
								save_files["scaler"] = scaler.state_dict()
							torch.save(save_files, best.format(epoch))
			print_log(f'{"=" * 15}epoch:{int(epoch)}--mAP:{round(coco_mAP, 4)}--the second stage{"=" * 15}', log)




if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='YOLO-Based')
	parser.add_argument('--first_epochs',default=6,type=int,help='train the first stage')
	parser.add_argument('--second_epochs',default=24,type=int,help='fine-tune')
	parser.add_argument('--start_epochs', default=0, type=int,help='continue training epochs')
	parser.add_argument('--batch_size', default=2, type=int)
	parser.add_argument('--save_best', default=False, type=bool,help='if save the best ckpt')
	parser.add_argument('--nc', default=5, type=int,help='number classes')
	parser.add_argument('--obj_loss', default='QualityFocal', type=str, help='balance loss for obj,eg [Focal,QualityFocal]')
	parser.add_argument('--cls_loss', default='GradientJointRepresentation', type=str, help='balance loss for cls eg [Focal,QualityFocal,EQ,GradientJointRepresentation]')
	parser.add_argument('--loc_loss', default='Iou', type=str, help='loss for location instances ex [Iou]')
	parser.add_argument('--weights', default="", type=str,help='reuse for continue training')
	parser.add_argument('--img_size', default=640, type=int, help='base size for multi-scale training')
	parser.add_argument('--multi_scale', default=True, type=bool, help='if multi-scale training')
	parser.add_argument('--data_root', default='./datasets/POCOCO', type=str, help='data root')
	parser.add_argument('--model_type', default='yolo', type=str, help='data type readout')
	parser.add_argument('--augment', default=True, type=bool, help='mosaic augment')
	parser.add_argument('--rect', default=False, type=bool, help='mosaic augment if false,else rect training')
	parser.add_argument('--backbone_name', default='Darknet', type=str, help='["VGG16","Convnext","Resnet50","SwinTransformer","DLA60","Darknet","CSPDarknet"]')
	parser.add_argument('--fpn_name', default='FeaturePyramidNetwork', type=str, help='["BackboneWithFPN","BackboneWithDyhead","BackboneWithPAFPN"] or ["FeaturePyramidNetwork","PAFPN","DyHead"] for TF-BASED')
	parser.add_argument('--freeze_layers', default=False, type=bool,help='freeze layers for transfer learning')
	parser.add_argument('--notest', default=False, type=bool, help='freeze layers for transfer learning')
	parser.add_argument('--amp', default=False, type=bool, help='Use torch.cuda.amp for mixed precision training')
	parser.add_argument('--name', default='', type=str, help='renames results.txt to results_name.txt if supplied')
	opt = parser.parse_args()
	tb_writer = SummaryWriter(comment=opt.name)
	print(opt)
	train(opt)












	import torch.nn as nn
	#动态加载独立模型
	# time_str = get_timestring()
	# backbone_name = 'DLA60'#[VGG16,Convnext,Resnet50,SwinTransformer]
	# fpn_name = 'BackboneWithFPN'#['BackboneWithPAFPN','BackboneWithFPN','BackboneWithDyhead'] for cnn framework; ['FeaturePyramidNetwork','PAFPN','DyHead'] for TF-based
	# for name,parms in model.named_parameters():
	# 	print(name)
	# layer_index = ['level3','level4','level5']
	# model = install_detector(backbone_name=backbone_name,fpn_name=fpn_name,nc=5)
	# for name, param in model.backbonewithfpn.body.named_parameters():
	# 	# print(name)
	# 	if all([not name.startswith(layer) for layer in layer_index]):
	# 		print(name)
	# model.train()
	# [print(x.shape) for x in model(torch.rand(1,3,224,224))]
	# names = ['backbone' in name for name, _ in model.named_modules()]
	# print(np.any(names))
	# for name,m in model.named_modules():
	# 	if 'backbone' in name:
	# 		print('1')
	# 		continue
	# for i,yolo_layer in enumerate(model.yolo_layers):
	# 	print(yolo_layer.anchor_vec)
	# model.train()
	# inp = torch.rand(1,3,224,224).to('cuda:0')


	#
	# pretrain_root = './pretrained/'
	# weights_path = [os.path.join(pretrain_root,x) for x in os.listdir(pretrain_root) if x.startswith(backbone_name.lower())]
	# ckpt = torch.load(weights_path[0])
	# print(weights_path[0])
	# print(ckpt.keys())
	# new_ckpt = OrderedDict()

	# for (k,v),(name,_) in zip(ckpt.items(),model.named_parameters()):
	# 	if k.split('.')[0] in name:
	# 		new_ckpt[name] = v
	# print(new_ckpt.keys())

	#匹配出需要冻结的层
	# print_log(f'{"*" * 15} {backbone_name} with {fpn_name} has been installed {"*" * 15}', log)
	# print_log("Time: {}".format(time_str), log)
	#
	# model.eval()
	# img = torch.rand(1,3,224,224)
	# x,p = model(img)
	# print(x.shape)
	# [print(x.shape) for x in p]

