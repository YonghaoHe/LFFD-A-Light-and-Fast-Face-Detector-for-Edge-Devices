# coding: utf-8
import sys
import os
import numpy
import numpy as np
import cv2
import time


# empty data batch class for dynamical properties
class DataBatch:
    pass


def NMS(boxes, overlap_threshold):
    '''

    :param boxes: numpy nx5, n is the number of boxes, 0:4->x1, y1, x2, y2, 4->score
    :param overlap_threshold:
    :return:
    '''
    if boxes.shape[0] == 0:
        return boxes

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype != numpy.float32:
        boxes = boxes.astype(numpy.float32)

    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    sc = boxes[:, 4]
    widths = x2 - x1
    heights = y2 - y1

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = heights * widths
    idxs = numpy.argsort(sc)  # 从小到大排序

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # compare secend highest score boxes
        xx1 = numpy.maximum(x1[i], x1[idxs[:last]])
        yy1 = numpy.maximum(y1[i], y1[idxs[:last]])
        xx2 = numpy.minimum(x2[i], x2[idxs[:last]])
        yy2 = numpy.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bo（ box
        w = numpy.maximum(0, xx2 - xx1 + 1)
        h = numpy.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = numpy.delete(idxs, numpy.concatenate(([last], numpy.where(overlap > overlap_threshold)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick]

class PredictCaffe(object):

    def __init__(self,
                 caffe,
                 symbol_file_path,
                 model_file_path,
                 receptive_field_list,
                 receptive_field_stride,
                 bbox_small_list,
                 bbox_large_list,
                 receptive_field_center_start,
                 num_output_scales
                 ):
        self.caffe = caffe
        self.deploy = symbol_file_path
        self.model = model_file_path

        self.receptive_field_list = receptive_field_list
        self.receptive_field_stride = receptive_field_stride
        self.bbox_small_list = bbox_small_list
        self.bbox_large_list = bbox_large_list
        self.receptive_field_center_start = receptive_field_center_start
        self.num_output_scales = num_output_scales
        self.constant = [i / 2.0 for i in self.receptive_field_list]
        self.input_height = 480
        self.input_width = 640
        self.__load_model()

    def __load_model(self):
        # load symbol and parameters
        print('----> load symbol file: %s\n----> load model file: %s' % (self.deploy, self.model))
        self.net = self.caffe.Net(self.deploy, self.model, self.caffe.TEST)
        print('----> Model is loaded successfully.')

    def predict(self, image, resize_scale=1, score_threshold=0.8, top_k=100, NMS_threshold=0.3, NMS_flag=True, skip_scale_branch_list=[]):

        if image.ndim != 3 or image.shape[2] != 3:
            print('Only RGB images are supported.')
            return None

        bbox_collection = []

        shorter_side = min(image.shape[:2])
        if shorter_side * resize_scale < 128:
            resize_scale = float(128) / shorter_side

        input_image = cv2.resize(image, (0, 0), fx=resize_scale, fy=resize_scale)

        input_image = input_image.astype(dtype=numpy.float32)
        input_image = input_image - 127.5
        input_image = input_image / 127.5
        input_image = input_image[:, :, :, numpy.newaxis]
        input_image = input_image.transpose([3, 2, 0, 1])
        print(input_image.shape)
        
        tic = time.time()
        self.net.blobs['data'].data[...] = input_image
        res = self.net.forward()
        outputs = []

        out_key = ['slice_axis20_1', 'conv8_3_bbox',
                   'slice_axis21_1', 'conv11_3_bbox',
                   'slice_axis22_1', 'conv14_3_bbox',
                   'slice_axis23_1', 'conv17_3_bbox',
                   'slice_axis24_1', 'conv20_3_bbox']
        for key in out_key:
            print(res[key].shape)
            outputs.append(res[key])
        toc = time.time()
        infer_time = (toc - tic) * 1000

        for i in range(self.num_output_scales):
            if i in skip_scale_branch_list:
                continue

            score_map = numpy.squeeze(outputs[i * 2], (0, 1))

            # score_map_show = score_map * 255
            # score_map_show[score_map_show < 0] = 0
            # score_map_show[score_map_show > 255] = 255
            # cv2.imshow('score_map' + str(i), cv2.resize(score_map_show.astype(dtype=numpy.uint8), (0, 0), fx=2, fy=2))
            # cv2.waitKey()

            bbox_map = numpy.squeeze(outputs[i * 2 + 1], 0)

            RF_center_Xs = numpy.array([self.receptive_field_center_start[i] + self.receptive_field_stride[i] * x for x in range(score_map.shape[1])])
            RF_center_Xs_mat = numpy.tile(RF_center_Xs, [score_map.shape[0], 1])
            RF_center_Ys = numpy.array([self.receptive_field_center_start[i] + self.receptive_field_stride[i] * y for y in range(score_map.shape[0])])
            RF_center_Ys_mat = numpy.tile(RF_center_Ys, [score_map.shape[1], 1]).T

            x_lt_mat = RF_center_Xs_mat - bbox_map[0, :, :] * self.constant[i]
            y_lt_mat = RF_center_Ys_mat - bbox_map[1, :, :] * self.constant[i]
            x_rb_mat = RF_center_Xs_mat - bbox_map[2, :, :] * self.constant[i]
            y_rb_mat = RF_center_Ys_mat - bbox_map[3, :, :] * self.constant[i]

            x_lt_mat = x_lt_mat / resize_scale
            x_lt_mat[x_lt_mat < 0] = 0
            y_lt_mat = y_lt_mat / resize_scale
            y_lt_mat[y_lt_mat < 0] = 0
            x_rb_mat = x_rb_mat / resize_scale
            x_rb_mat[x_rb_mat > image.shape[1]] = image.shape[1]
            y_rb_mat = y_rb_mat / resize_scale
            y_rb_mat[y_rb_mat > image.shape[0]] = image.shape[0]

            select_index = numpy.where(score_map > score_threshold)
            for idx in range(select_index[0].size):
                bbox_collection.append((x_lt_mat[select_index[0][idx], select_index[1][idx]],
                                        y_lt_mat[select_index[0][idx], select_index[1][idx]],
                                        x_rb_mat[select_index[0][idx], select_index[1][idx]],
                                        y_rb_mat[select_index[0][idx], select_index[1][idx]],
                                        score_map[select_index[0][idx], select_index[1][idx]]))

        # NMS
        bbox_collection = sorted(bbox_collection, key=lambda item: item[-1], reverse=True)
        if len(bbox_collection) > top_k:
            bbox_collection = bbox_collection[0:top_k]
        bbox_collection_numpy = numpy.array(bbox_collection, dtype=numpy.float32)

        if NMS_flag:
            final_bboxes = NMS(bbox_collection_numpy, NMS_threshold)
            final_bboxes_ = []
            for i in range(final_bboxes.shape[0]):
                final_bboxes_.append((final_bboxes[i, 0], final_bboxes[i, 1], final_bboxes[i, 2], final_bboxes[i, 3], final_bboxes[i, 4]))

            return final_bboxes_, infer_time
        else:
            return bbox_collection_numpy, infer_time


def run_prediction_folder():
    sys.path.append('..')
    from config_farm import configuration_10_320_20L_5scales_v2 as cfg
    import caffe

    debug_folder = '' # fill the folder that contains images
    file_name_list = [file_name for file_name in os.listdir(debug_folder) if file_name.lower().endswith('jpg')]

    symbol_file_path = './configuration_10_320_20L_5scales_v2/symbol_10_320_20L_5scales_v2_deploy.prototxt'
    model_file_path = './configuration_10_320_20L_5scales_v2/train_10_320_20L_5scales_v2_iter_1000000.caffemodel'
    my_predictor = PredictCaffe(caffe=caffe,
                           symbol_file_path=symbol_file_path,
                           model_file_path=model_file_path,
                           receptive_field_list=cfg.param_receptive_field_list,
                           receptive_field_stride=cfg.param_receptive_field_stride,
                           bbox_small_list=cfg.param_bbox_small_list,
                           bbox_large_list=cfg.param_bbox_large_list,
                           receptive_field_center_start=cfg.param_receptive_field_center_start,
                           num_output_scales=cfg.param_num_output_scales)

    for file_name in file_name_list:
        im = cv2.imread(os.path.join(debug_folder, file_name))

        bboxes, _ = my_predictor.predict(im, resize_scale=1, score_threshold=0.3, top_k=10000, NMS_threshold=0.3, NMS_flag=True, skip_scale_branch_list=[])
        for bbox in bboxes:
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        if max(im.shape[:2]) > 1600:
            scale = 1600/max(im.shape[:2])
            im = cv2.resize(im, (0, 0), fx=scale, fy=scale)
        cv2.imshow('im', im)
        cv2.waitKey()
        # cv2.imwrite(os.path.join(debug_folder, file_name.replace('.jpg','_result.jpg')), im)


if __name__ == '__main__':
    run_prediction_folder()
