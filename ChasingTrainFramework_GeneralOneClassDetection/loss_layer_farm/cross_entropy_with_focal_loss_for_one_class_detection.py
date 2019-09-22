# -*- coding: utf-8 -*-
# @date         : 19-1-23
# @author       : MindBreaker
# @module       :

import mxnet as mx
import numpy as np
import logging


class focal_loss_for_twoclass(mx.operator.CustomOp):
    '''
    1, the in_data[0], namely the pred, must be applied with softmax before running this loss operator
    2, this CE operator is only for two-class situation, the 0-index indicates pos(foreground), and the 1-index is for neg(background)
    '''

    def __init__(self, alpha=0.25, gamma=2):
        super(focal_loss_for_twoclass, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, is_train, req, in_data, out_data, aux):
        pred = in_data[0]
        label = in_data[1]
        pred_softmax = mx.ndarray.softmax(pred, axis=1)
        pred_log = mx.ndarray.log(pred_softmax)
        cross_entropy = - label * pred_log

        self.assign(out_data[0], req[0], cross_entropy)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        pred = in_data[0]
        label = in_data[1]
        mask = in_data[2]

        pred_softmax = mx.ndarray.softmax(pred, axis=1)

        # print('pos mean prob:', mx.ndarray.mean(pred_softmax[:, 0, :, :][label[:, 0, :, :] > 0.5]).asnumpy())
        # print('neg mean prob:', mx.ndarray.mean(pred_softmax[:, 1, :, :][label[:, 1, :, :] > 0.5]).asnumpy())

        # pos_flag = label[:, 0, :, :] > 0.5
        # neg_flag = label[:, 1, :, :] > 0.5

        FL_gradient = -self.gamma * mx.ndarray.power(1 - pred_softmax, self.gamma - 1) * mx.ndarray.log(pred_softmax) * pred_softmax + mx.ndarray.power(1 - pred_softmax, self.gamma)

        FL_gradient[:, 0, :, :] *= self.alpha
        FL_gradient[:, 1, :, :] *= 1 - self.alpha

        FL_gradient *= (pred_softmax-label)

        FL_gradient /= mx.ndarray.sum(mask).asnumpy()[0]
        # print('mean grad:', mx.ndarray.mean(mx.ndarray.abs(FL_gradient)).asnumpy())

        self.assign(in_grad[0], req[0], FL_gradient)


@mx.operator.register("focal_loss_for_twoclass")
class focal_loss_for_twoclass_Prop(mx.operator.CustomOpProp):
    def __init__(self):
        super(focal_loss_for_twoclass_Prop, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['pred', 'label', 'mask']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = in_shape[0]
        mask_shape = in_shape[0]
        output_shape = in_shape[0]
        return [data_shape, label_shape, mask_shape], [output_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return focal_loss_for_twoclass()
