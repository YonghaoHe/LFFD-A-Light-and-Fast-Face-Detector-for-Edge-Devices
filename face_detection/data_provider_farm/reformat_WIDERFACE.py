# -*- coding: utf-8 -*-

import cv2
import os
import numpy


def generate_data_list():
    root = '/media/heyonghao/HYH-4T-WD/public_dataset/WIDER_FACE/WIDER_val/images'
    orig_txt_path = '/media/heyonghao/HYH-4T-WD/public_dataset/WIDER_FACE/wider_face_split/wider_face_val_bbx_gt.txt'
    save_txt_path = './data_folder/val_pos_list.txt'

    fin = open(orig_txt_path, 'r')
    fout = open(save_txt_path, 'w')
    fin_line = fin.readline()
    fout_line = ''
    counter = 0
    while fin_line:
        fin_line = fin_line.strip('\n')
        if fin_line.endswith('.jpg'):
            print('processing %s' % fin_line)
            file_path = os.path.join(root, fin_line)
            fout_line += file_path
            fin_line = fin.readline()
            continue
        # print(fin_line)
        num_bboxes = int(fin_line)

        bbox_list = []
        for n in range(num_bboxes):
            fin_line = fin.readline()
            fin_line = fin_line.strip(' \n').split(' ')
            x1 = int(fin_line[0])
            y1 = int(fin_line[1])
            width = int(fin_line[2])
            height = int(fin_line[3])
            blur = int(fin_line[4])
            expression = int(fin_line[5])
            illumination = int(fin_line[6])
            invalid = int(fin_line[7])
            occlusion = int(fin_line[8])
            pose = int(fin_line[9])
            if x1 <= 0 or y1 <= 0 or width <= 0 or height <= 0 or invalid == 1 or max(width, height) < 10:
                continue
            bbox_list.append((x1, y1, width, height))

        if len(bbox_list) == 0:
            fout_line += ',0,0\n'
        else:
            fout_line += ',1,' + str(len(bbox_list))
            for bbox in bbox_list:
                fout_line += ',' + str(bbox[0]) + ',' + str(bbox[1]) + ',' + str(bbox[2]) + ',' + str(bbox[3])
            fout_line += '\n'
        fout.write(fout_line)
        counter += len(bbox_list)

        fout_line = ''
        if num_bboxes != 0:
            fin_line = fin.readline()
        else:
            _ = fin.readline()
            fin_line = fin.readline()
    print(counter)
    fin.close()
    fout.close()


def generate_neg_image():
    train_list_file_path = './data_folder/train_pos_list.txt'
    train_neg_list_file_path = './data_folder/train_neg_list.txt'
    neg_save_root = '/media/heyonghao/HYH-4T-WD/public_dataset/WIDER_FACE/WIDER_train/neg_images'
    if not os.path.exists(neg_save_root):
        os.makedirs(neg_save_root)

    fin = open(train_list_file_path, 'r')
    fout = open(train_neg_list_file_path, 'w')

    crop_threshold = 100

    counter = 0
    for line in fin:
        line = line.strip('\n').split(',')
        num_bboxes = int(line[2])
        if num_bboxes == 0:
            continue

        im = cv2.imread(line[0])

        bboxes = numpy.zeros((num_bboxes, 4), dtype=numpy.int32)
        for n in range(num_bboxes):
            x = int(line[3 + n * 4])
            y = int(line[3 + n * 4 + 1])
            width = int(line[3 + n * 4 + 2])
            height = int(line[3 + n * 4 + 3])
            bboxes[n, :] = [x, y, width, height]

        left_boundary = int(numpy.min(bboxes[:, 0]))
        right_boundary = int(numpy.max(bboxes[:, 0] + bboxes[:, 2]))
        top_boundary = int(numpy.min(bboxes[:, 1]))
        bottom_boundary = int(numpy.max(bboxes[:, 1] + bboxes[:, 3]))
        if left_boundary >= crop_threshold:
            im_left = im[:, :left_boundary, :]
            cv2.imwrite(os.path.join(neg_save_root, str(counter) + '.jpg'), im_left)

            fout.write(os.path.join(neg_save_root, str(counter) + '.jpg') + ',0,0\n')
            print('A new bg im saved. %d' % counter)
            counter += 1
        if im.shape[1] - right_boundary >= crop_threshold:
            im_right = im[:, right_boundary:, :]
            cv2.imwrite(os.path.join(neg_save_root, str(counter) + '.jpg'), im_right)

            fout.write(os.path.join(neg_save_root, str(counter) + '.jpg') + ',0,0\n')
            print('A new bg im saved. %d' % counter)
            counter += 1
        if top_boundary >= crop_threshold:
            im_top = im[:top_boundary, :, :]
            cv2.imwrite(os.path.join(neg_save_root, str(counter) + '.jpg'), im_top)

            fout.write(os.path.join(neg_save_root, str(counter) + '.jpg') + ',0,0\n')
            print('A new bg im saved. %d' % counter)
            counter += 1
        if im.shape[0] - bottom_boundary >= crop_threshold:
            im_bottom = im[bottom_boundary:, :, :]
            cv2.imwrite(os.path.join(neg_save_root, str(counter) + '.jpg'), im_bottom)

            fout.write(os.path.join(neg_save_root, str(counter) + '.jpg') + ',0,0\n')
            print('A new bg im saved. %d' % counter)
            counter += 1

    fin.close()
    fout.close()


def merge_list():
    train_pos_list_file_path = './data_folder/train_pos_list.txt'
    train_neg_list_file_path = './data_folder/train_neg_list.txt'
    train_list_file_path = './data_folder/train_list.txt'

    fin1 = open(train_pos_list_file_path, 'r')
    fin2 = open(train_neg_list_file_path, 'r')
    fout = open(train_list_file_path, 'w')
    for line in fin1:
        fout.write(line)
    for line in fin2:
        fout.write(line)

    fin1.close()
    fin2.close()
    fout.close()


def check_txt():
    txt_path = './data_folder/val_pos_list.txt'
    fin = open(txt_path, 'r')
    lines = fin.readlines()
    fin.close()
    import random
    random.shuffle(lines)
    for idx, line in enumerate(lines):
        line = line.strip(' \n').split(',')
        num_bboxes = int(line[2])
        if num_bboxes == 0:
            continue
        im = cv2.imread(line[0])
        for n in range(num_bboxes):
            x = int(line[3 + n * 4])
            y = int(line[3 + n * 4 + 1])
            width = int(line[3 + n * 4 + 2])
            height = int(line[3 + n * 4 + 3])
            cv2.rectangle(im, (x, y), (x + width, y + height), (255, 255, 0), 2)
        cv2.imshow('img', im)
        cv2.waitKey()


if __name__ == '__main__':
    # generate_data_list()
    # generate_neg_image()
    # merge_list()
    check_txt()
