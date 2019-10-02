# -*- coding: utf-8 -*-
import os
import sys
import cv2
import math
import re

sys.path.append('..')
# change the config as your need
from config_farm import configuration_64_512_16L_3scales_v1 as cfg
import mxnet
from predict import Predict


def generate_gt_files():
    txt_file_path = '../data_provider_farm/data_folder/data_list_CCPD_test.txt'
    gt_file_root = './CCPD_testset_gt_files_for_evaluation'

    if not os.path.exists(gt_file_root):
        os.makedirs(gt_file_root)

    fin = open(txt_file_path, 'r')

    counter = 0
    for line in fin:
        line = line.strip('\n').split(',')
        im_path = os.path.basename(line[0])
        num_bboxes = int(line[2])
        if num_bboxes == 0:
            continue
        bbox_list = []
        for i in range(num_bboxes):
            xmin = int(float(line[3+i*4]))
            ymin = int(float(line[4+i*4]))
            width = int(float(line[5+i*4]))
            height = int(float(line[6+i*4]))
            bbox_list.append((xmin, ymin, width, height))

        gt_file_name = im_path.replace('jpg', 'txt')

        fout = open(os.path.join(gt_file_root, gt_file_name), 'w')
        for bbox in bbox_list:
            line_str = 'LP ' + str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[2]) + ' ' + str(bbox[3])
            fout.write(line_str + '\n')
        fout.close()
        counter += 1
        print(counter)
    fin.close()


def generate_predicted_files():
    # set the proper symbol file and model file
    symbol_file_path = '../symbol_farm/symbol_64_512_16L_3scales_v1_deploy.json'
    model_file_path = '../saved_model/configuration_64_512_16L_3scales_v1_2019-09-29-13-41-44/train_64_512_16L_3scales_v1_iter_600000.params'
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
    txt_file_path = '../data_provider_farm/data_folder/data_list_CCPD_test.txt'
    predicted_file_root = './CCPD_testset_predicted_files_for_evaluation_' + os.path.basename(model_file_path).split('.')[0]

    if not os.path.exists(predicted_file_root):
        os.makedirs(predicted_file_root)

    fin = open(txt_file_path, 'r')

    resize_scale = 1
    score_threshold = 0.2
    NMS_threshold = 0.6
    counter = 0

    for line in fin:
        line = line.strip('\n').split(',')

        im = cv2.imread(line[0], cv2.IMREAD_COLOR)

        bboxes = my_predictor.predict(im, resize_scale=resize_scale, score_threshold=score_threshold, top_k=10000, NMS_threshold=NMS_threshold)

        # for bbox in bboxes:
        #     cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 1)
        # cv2.imshow('im',im)
        # cv2.waitKey()
        predicted_file_name = os.path.basename(line[0]).replace('jpg', 'txt')
        fout = open(os.path.join(predicted_file_root, predicted_file_name), 'w')
        for bbox in bboxes:
            fout.write('LP %.03f %d %d %d %d' % (bbox[4] if bbox[4] <= 1 else 1, math.floor(bbox[0]), math.floor(bbox[1]), math.ceil(bbox[2] - bbox[0]), math.ceil(bbox[3] - bbox[1])) + '\n')
        fout.close()
        counter += 1
        print('[%d] is processed.' % counter)


if __name__ == '__main__':
    # generate_gt_files()
    generate_predicted_files()
