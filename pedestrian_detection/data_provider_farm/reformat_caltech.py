import numpy
import os
import cv2
import json
import math


def generate_data_list():
    annotation_root = '/media/heyonghao/HYH-4T-WD/public_dataset/Caltech/Caltech_new_annotations/anno_test_1xnew'
    image_root = '/media/heyonghao/HYH-4T-WD/public_dataset/Caltech/Caltech_data/extracted_data'

    list_file_path = './data_folder/data_list_caltech_test.txt'
    if not os.path.exists(os.path.dirname(list_file_path)):
        os.makedirs(os.path.dirname(list_file_path))
    fout = open(list_file_path, 'w')

    counter = 0
    for parent, dirnames, filenames in os.walk(annotation_root):
        for filename in filenames:
            if not filename.endswith('.txt'):
                continue

            filename_splits = filename[:-4].split('_')
            set_name = filename_splits[0]
            seq_name = filename_splits[1]
            img_name = filename_splits[2]

            img_path = os.path.join(image_root, set_name, seq_name, 'images', img_name)
            if not os.path.exists(img_path):
                print('The corresponding image does not exist! [%s]' % img_path)
                continue

            line = img_path

            fin_anno = open(os.path.join(parent, filename), 'r')

            bbox_list = []
            for i, anno in enumerate(fin_anno):
                if i == 0:
                    continue
                anno = anno.strip('\n').split(' ')
                if anno[0] != 'person':
                    continue
                x = math.floor(float(anno[1]))
                y = math.floor(float(anno[2]))
                width = math.ceil(float(anno[3]))
                height = math.ceil(float(anno[4]))

                width_vis = math.ceil(float(anno[8]))
                height_vis = math.ceil(float(anno[9]))

                if (width_vis*height_vis)/(width*height) < 0.2:
                    continue

                bbox_list.append((x, y, width, height))
            if len(bbox_list) == 0:
                line += ',0,0'
                fout.write(line + '\n')
            else:
                bbox_line = ''
                for bbox in bbox_list:
                    bbox_line += ',' + str(bbox[0]) + ',' + str(bbox[1]) + ',' + str(bbox[2]) + ',' + str(bbox[3])
                line += ',1,' + str(len(bbox_list)) + bbox_line
                fout.write(line + '\n')
            counter += 1
            print(counter)

    fout.close()


def show_image():
    list_file_path = './data_folder/data_list_caltech_test.txt'

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
    list_file_path = './data_folder/data_list_caltech_test.txt'

    fin = open(list_file_path, 'r')
    lines = fin.readlines()
    fin.close()

    bin_size = 10
    longer_bin_dict = {}
    shorter_bin_dict = {}
    counter_pos = 0
    counter_neg = 0
    for line in lines:
        line = line.strip('\n').split(',')
        if line[1] == '0':
            counter_neg += 1
            continue
        else:
            counter_pos += 1
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

    total_pedestrian = 0
    print('shorter side based statistics:')
    shorter_bin_dict_key_list = sorted(shorter_bin_dict)
    for k in shorter_bin_dict_key_list:
        v = shorter_bin_dict[k]
        total_pedestrian += v
        print('[%d-%d): %d' % (k * bin_size, k * bin_size + bin_size, v))

    print('longer side based statistics:')
    longer_bin_dict_key_list = sorted(longer_bin_dict)
    for k in longer_bin_dict_key_list:
        v = longer_bin_dict[k]
        print('[%d-%d): %d' % (k * bin_size, k * bin_size + bin_size, v))

    print('num pos: %d, num neg: %d' % (counter_pos, counter_neg))
    print('total pedestrian: %d' % total_pedestrian)


if __name__ == '__main__':
    # generate_data_list()
    show_image()
    # dataset_statistics()
