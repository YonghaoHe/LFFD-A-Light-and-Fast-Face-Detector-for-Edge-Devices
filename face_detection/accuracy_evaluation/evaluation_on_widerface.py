# -*- coding: utf-8 -*-
import os
import sys
import cv2
import math
sys.path.append('..')
# change the config as your need
from config_farm import configuration_10_320_20L_5scales_v2 as cfg
import mxnet
from predict import Predict

# set the proper symbol file and model file
symbol_file_path = '../symbol_farm/symbol_10_320_20L_5scales_v2_deploy.json'
model_file_path = '../saved_model/configuration_10_320_20L_5scales_v2/train_10_320_20L_5scales_v2_iter_1800000.params'
my_predictor = Predict(mxnet=mxnet,
                       symbol_file_path=symbol_file_path,
                       model_file_path=model_file_path,
                       ctx=mxnet.gpu(0),
                       receptive_field_list=cfg.param_receptive_field_list,
                       receptive_field_stride=cfg.param_receptive_field_stride,
                       bbox_small_list=cfg.param_bbox_small_list,
                       bbox_large_list=cfg.param_bbox_large_list,
                       receptive_field_center_start=cfg.param_receptive_field_center_start,
                       num_output_scales=cfg.param_num_output_scales)

# set the val root, the path should look like XXXX/WIDER_val/images
val_image_root = 'XXXX/WIDER_val/images'
val_result_txt_save_root = './widerface_val_' + os.path.basename(model_file_path).split('.')[0] + '_result_txt/'
if not os.path.exists(val_result_txt_save_root):
    os.makedirs(val_result_txt_save_root)

resize_scale = 1
score_threshold = 0.11
NMS_threshold = 0.4
counter = 0
for parent, dir_names, file_names in os.walk(val_image_root):
    for file_name in file_names:
        if not file_name.lower().endswith('jpg'):
            continue

        im = cv2.imread(os.path.join(parent, file_name), cv2.IMREAD_COLOR)

        bboxes = my_predictor.predict(im, resize_scale=resize_scale, score_threshold=score_threshold, top_k=10000, NMS_threshold=NMS_threshold)

        # for bbox in bboxes:
        #     cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 1)
        # cv2.imshow('im',im)
        # cv2.waitKey()

        event_name = parent.split('/')[-1]
        if not os.path.exists(os.path.join(val_result_txt_save_root, event_name)):
            os.makedirs(os.path.join(val_result_txt_save_root, event_name))
        fout = open(os.path.join(val_result_txt_save_root, event_name, file_name.split('.')[0] + '.txt'), 'w')
        fout.write(file_name.split('.')[0] + '\n')
        fout.write(str(len(bboxes)) + '\n')
        for bbox in bboxes:
            fout.write('%d %d %d %d %.03f' % (math.floor(bbox[0]), math.floor(bbox[1]), math.ceil(bbox[2] - bbox[0]), math.ceil(bbox[3] - bbox[1]), bbox[4] if bbox[4] <= 1 else 1) + '\n')
        fout.close()
        counter += 1
        print('[%d] %s is processed.' % (counter, file_name))


