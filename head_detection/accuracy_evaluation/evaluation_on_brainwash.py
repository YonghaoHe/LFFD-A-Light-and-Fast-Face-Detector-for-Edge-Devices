# -*- coding: utf-8 -*-
import os
import sys
import cv2
import math
import re

sys.path.append('..')
# change the config as your need
from config_farm import configuration_10_160_17L_4scales_v1 as cfg
import mxnet
from predict import Predict


def generate_gt_files():
    txt_file_path = '/media/heyonghao/HYH-4T-WD/public_dataset/head_detection/brainwash/brainwash/brainwash_test.idl'
    gt_file_root = './brainwash_testset_gt_files_for_evaluation'

    if not os.path.exists(gt_file_root):
        os.makedirs(gt_file_root)

    fin = open(txt_file_path, 'r')

    counter = 0
    for line in fin:
        line = line.strip(';\n')
        im_path = re.findall('["](.*?)["]', line)[0]

        bbox_str_list = re.findall('[(](.*?)[)]', line)
        bbox_list = []
        for bbox_str in bbox_str_list:
            bbox_str = bbox_str.split(', ')
            xmin = int(float(bbox_str[0]))
            ymin = int(float(bbox_str[1]))
            xmax = int(float(bbox_str[2]))
            ymax = int(float(bbox_str[3]))
            bbox_list.append((xmin, ymin, xmax - xmin + 1, ymax - ymin + 1))

        if len(bbox_list) != 0:
            gt_file_name = im_path.replace('/', '_')
            gt_file_name = gt_file_name.replace('png', 'txt')
            fout = open(os.path.join(gt_file_root, gt_file_name), 'w')
            for bbox in bbox_list:
                line_str = 'head ' + str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[2]) + ' ' + str(bbox[3])
                fout.write(line_str + '\n')
            fout.close()
            counter += 1
            print(counter)
    fin.close()


def generate_predicted_files():
    # set the proper symbol file and model file
    symbol_file_path = '../symbol_farm/symbol_10_160_17L_4scales_v1_deploy.json'
    model_file_path = '../saved_model/configuration_10_160_17L_4scales_v1_2019-09-20-13-08-26/train_10_160_17L_4scales_v1_iter_800000.params'
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
    txt_file_path = '/media/heyonghao/HYH-4T-WD/public_dataset/head_detection/brainwash/brainwash/brainwash_test.idl'
    image_root = '/media/heyonghao/HYH-4T-WD/public_dataset/head_detection/brainwash/brainwash'
    predicted_file_root = './brainwash_testset_predicted_files_for_evaluation_' + os.path.basename(model_file_path).split('.')[0]

    if not os.path.exists(predicted_file_root):
        os.makedirs(predicted_file_root)

    fin = open(txt_file_path, 'r')

    resize_scale = 1
    score_threshold = 0.05
    NMS_threshold = 0.6
    counter = 0

    for line in fin:
        line = line.strip(';\n')
        im_path = re.findall('["](.*?)["]', line)[0]

        im = cv2.imread(os.path.join(image_root, im_path), cv2.IMREAD_COLOR)

        bboxes = my_predictor.predict(im, resize_scale=resize_scale, score_threshold=score_threshold, top_k=10000, NMS_threshold=NMS_threshold)

        # for bbox in bboxes:
        #     cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 1)
        # cv2.imshow('im',im)
        # cv2.waitKey()
        predicted_file_name = im_path.replace('/', '_')
        predicted_file_name = predicted_file_name.replace('png', 'txt')
        fout = open(os.path.join(predicted_file_root, predicted_file_name), 'w')
        for bbox in bboxes:
            fout.write('head %.03f %d %d %d %d' % (bbox[4] if bbox[4] <= 1 else 1, math.floor(bbox[0]), math.floor(bbox[1]), math.ceil(bbox[2] - bbox[0]), math.ceil(bbox[3] - bbox[1])) + '\n')
        fout.close()
        counter += 1
        print('[%d] is processed.' % counter)


if __name__ == '__main__':
    # generate_gt_files()
    generate_predicted_files()
