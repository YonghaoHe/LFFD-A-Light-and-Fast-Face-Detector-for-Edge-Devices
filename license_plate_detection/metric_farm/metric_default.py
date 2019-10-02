# -*- coding: utf-8 -*-

import numpy
import mxnet


class Metric:
    # 需要输入多少个loss，即scale个数
    def __init__(self, num_scales):
        self.sum_metric = [0.0 for i in range(num_scales * 2)]
        self.num_update = 0
        self.num_scales = num_scales
        self.num_nonzero = [1.0 for i in range(num_scales * 2)]
        self.scale_factor = 10000

    # it is expected that the shape is num*c*h*w
    def update(self, labels, preds):  # 这里需要注意label里面item的顺序。要参考prefetching_dataiter

        for i in range(self.num_scales):
            mask = labels[i * 2]  # 先mask
            label = labels[i * 2 + 1]  # 后label

            score_mask = mxnet.ndarray.slice_axis(mask, axis=1, begin=0, end=2).asnumpy()
            bbox_mask = mxnet.ndarray.slice_axis(mask, axis=1, begin=2, end=6).asnumpy()

            label_bbox = mxnet.ndarray.slice_axis(label, axis=1, begin=2, end=6).asnumpy()

            pred_score = preds[i * 2].asnumpy()
            pred_bbox = preds[i * 2 + 1].asnumpy()

            loss_score = numpy.sum(pred_score * score_mask)
            loss_bbox = numpy.sum((label_bbox - pred_bbox) ** 2.0)

            self.num_nonzero[i * 2] += numpy.sum(score_mask[:, 0, :, :] > 0.5)
            self.num_nonzero[i * 2 + 1] += numpy.sum(bbox_mask > 0.5)
            self.sum_metric[i * 2] += loss_score
            self.sum_metric[i * 2 + 1] += loss_bbox

        self.num_update += 1

    def get(self):
        return_string_list = []
        for i in range(self.num_scales):
            return_string_list.append('CE_loss_score_' + str(i))
            return_string_list.append('SE_loss_bbox_' + str(i))

        return return_string_list, [m / self.num_nonzero[i] * self.scale_factor for i, m in enumerate(self.sum_metric)]

    def reset(self):
        self.sum_metric = [0.0 for i in range(self.num_scales * 2)]
        self.num_update = 0
        self.num_nonzero = [1.0 for i in range(self.num_scales * 2)]
