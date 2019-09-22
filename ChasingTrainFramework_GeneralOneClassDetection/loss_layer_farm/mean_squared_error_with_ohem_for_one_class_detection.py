# -*- coding: utf-8 -*-
'''
squared error with online hard example mining
'''
import mxnet as mx


class mean_squared_error_with_ohem_for_one_class_detection(mx.operator.CustomOp):
    def __init__(self, ohem_ratio):
        super(mean_squared_error_with_ohem_for_one_class_detection, self).__init__()
        self.ohem_ratio = ohem_ratio

    def forward(self, is_train, req, in_data, out_data, aux):
        pred = in_data[0]
        self.assign(out_data[0], req[0], pred)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        pred = out_data[0]
        label = in_data[1]
        loss = pred - label

        # perform OHEM
        num_select = int(label.size * self.ohem_ratio)
        loss_abs = mx.nd.abs(loss)
        loss_sort = mx.nd.sort(loss_abs.reshape((1, -1)), is_ascend=False)
        min_threshold = loss_sort[0][num_select].asnumpy()[0]
        select_flag = loss_abs >= min_threshold
        loss *= select_flag
        loss /= num_select

        self.assign(in_grad[0], req[0], loss)


@mx.operator.register("mean_squared_error_with_ohem_for_one_class_detection")
class mean_squared_error_with_ohem_for_one_class_detection_Prop(mx.operator.CustomOpProp):
    def __init__(self, ohem_ratio=0.25):
        super(mean_squared_error_with_ohem_for_one_class_detection_Prop, self).__init__(need_top_grad=False)
        self.ohem_ratio = ohem_ratio

    def list_arguments(self):
        return ['pred', 'label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        pred_shape = in_shape[0]
        label_shape = in_shape[0]
        output_shape = in_shape[0]
        return [pred_shape, label_shape], [output_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return mean_squared_error_with_ohem_for_one_class_detection(self.ohem_ratio)
