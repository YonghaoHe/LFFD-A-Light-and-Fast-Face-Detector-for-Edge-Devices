import numpy
import os
import cv2
import json
import math
import re

'''
'''


def generate_data_list():
    txt_file_path = '/media/heyonghao/HYH-4T-WD/public_dataset/head_detection/brainwash/brainwash/brainwash_test.idl'
    image_root = '/media/heyonghao/HYH-4T-WD/public_dataset/head_detection/brainwash/brainwash'

    list_file_path = './data_folder/data_list_brainwash_test.txt'
    if not os.path.exists(os.path.dirname(list_file_path)):
        os.makedirs(os.path.dirname(list_file_path))
    fin = open(txt_file_path, 'r')
    fout = open(list_file_path, 'w')

    counter = 0
    for line in fin:
        line = line.strip(';\n')
        im_path = re.findall('["](.*?)["]', line)[0]
        im_path = os.path.join(image_root, im_path)
        if not os.path.exists(im_path):
            print('im file does not exist : %s'%im_path)
            continue
        bbox_str_list = re.findall('[(](.*?)[)]', line)
        bbox_list = []
        for bbox_str in bbox_str_list:
            bbox_str = bbox_str.split(', ')
            xmin = int(float(bbox_str[0]))
            ymin = int(float(bbox_str[1]))
            xmax = int(float(bbox_str[2]))
            ymax = int(float(bbox_str[3]))
            bbox_list.append((xmin, ymin, xmax-xmin+1, ymax-ymin+1))

        if len(bbox_list) == 0:
            line_str = im_path+',0,0'
            fout.write(line_str+'\n')
        else:
            line_str = im_path+',1,'+str(len(bbox_list))
            for bbox in bbox_list:
                line_str += ','+str(bbox[0])+','+str(bbox[1])+','+str(bbox[2])+','+str(bbox[3])
            fout.write(line_str + '\n')
        counter += 1
        print(counter)

    fout.close()
    fin.close()


def show_image():
    list_file_path = './data_folder/data_list_brainwash_test.txt'

    fin = open(list_file_path, 'r')
    lines = fin.readlines()
    fin.close()

    import random
    random.shuffle(lines)
    for line in lines:
        line = line.strip('\n').split(',')

        im = cv2.imread(line[0])

        bboxes = []
        num_bboxes = int(line[2])
        for i in range(num_bboxes):
            xmin = int(line[3 + i * 4])
            ymin = int(line[4 + i * 4])
            width = int(line[5 + i * 4])
            height = int(line[6 + i * 4])
            bboxes.append((xmin, ymin, xmin + width - 1, ymin + height - 1))

        for bbox in bboxes:
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 2)

        cv2.imshow('im', im)
        cv2.waitKey()


def dataset_statistics():
    list_file_path = './data_folder/data_list_brainwash_test.txt'

    fin = open(list_file_path, 'r')
    lines = fin.readlines()
    fin.close()

    bin_size = 5
    longer_bin_dict = {}
    shorter_bin_dict = {}
    for line in lines:
        line = line.strip('\n').split(',')
        num_bboxes = int(line[2])
        for i in range(num_bboxes):
            width = int(line[5 + i * 4])
            height = int(line[6 + i * 4])

            longer_side = max(width, height)
            shorter_side = min(width, height)

            key = int(longer_side / bin_size)
            if key in longer_bin_dict:
                longer_bin_dict[key] += 1
            else:
                longer_bin_dict[key] = 1

            key = int(shorter_side / bin_size)
            if key in shorter_bin_dict:
                shorter_bin_dict[key] += 1
            else:
                shorter_bin_dict[key] = 1

    print('shorter side based statistics:')
    shorter_bin_dict_key_list = sorted(shorter_bin_dict)
    for k in shorter_bin_dict_key_list:
        v = shorter_bin_dict[k]
        print('[%d-%d): %d' % (k * bin_size, k * bin_size + bin_size, v))

    print('longer side based statistics:')
    longer_bin_dict_key_list = sorted(longer_bin_dict)
    for k in longer_bin_dict_key_list:
        v = longer_bin_dict[k]
        print('[%d-%d): %d' % (k * bin_size, k * bin_size + bin_size, v))


if __name__ == '__main__':
    # generate_data_list()
    # show_image()
    dataset_statistics()

