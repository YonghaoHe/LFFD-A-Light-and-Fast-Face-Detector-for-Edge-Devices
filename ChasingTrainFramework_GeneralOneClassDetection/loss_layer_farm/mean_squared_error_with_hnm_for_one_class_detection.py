# -*- coding: utf-8 -*-
'''
squared error with hard negative mining
'''
import mxnet as mx


class mean_squared_error_with_hnm_for_one_class_detection(mx.operator.CustomOp):
    def __init__(self, hnm_ratio):
        super(mean_squared_error_with_hnm_for_one_class_detection, self).__init__()
        self.hnm_ratio = int(hnm_ratio)

    def forward(self, is_train, req, in_data, out_data, aux):
        pred = in_data[0]
        self.assign(out_data[0], req[0], pred)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        pred = in_data[0]
        label = in_data[1]
        loss = pred - label  # Standard gradient in MXNET for Regression loss.
        if self.hnm_ratio != 0:
            pos_flag = (label > 0)
            pos_num = mx.ndarray.sum(pos_flag).asnumpy()[0]  # 得到正样本的个数
            if pos_num > 0:
                neg_flag = (label < 0.0001)
                neg_num = mx.ndarray.sum(neg_flag).asnumpy()[0]
                neg_num_selected = min(int(self.hnm_ratio * pos_num), int(neg_num))
                neg_loss = mx.ndarray.abs(loss * neg_flag)  # non-negative value
                neg_loss_tem = mx.ndarray.sort(neg_loss.reshape((1, -1)), is_ascend=False)

                top_loss_min = neg_loss_tem[0][neg_num_selected].asnumpy()[0]
                neg_loss_flag = (neg_loss >= top_loss_min)
                loss_mask = mx.ndarray.logical_or(neg_loss_flag, pos_flag)
            else:
                neg_choice_ratio = 0.1
                neg_num_selected = int(loss.size * neg_choice_ratio)
                loss_abs = mx.ndarray.abs(loss)
                neg_loss_tem = mx.ndarray.sort(loss_abs.reshape((1, -1)), is_ascend=False)
                top_loss_min = neg_loss_tem[0][neg_num_selected].asnumpy()[0]
                # logging.info('top_loss_min:%0.4f', top_loss_min)
                loss_mask = (loss_abs >= top_loss_min)

            # logging.info('remained_num:%d', mx.ndarray.sum(mask).asnumpy()[0])

            loss *= loss_mask
        loss /= loss[0].size
        self.assign(in_grad[0], req[0], loss)


@mx.operator.register("mean_squared_error_with_hnm_for_one_class_detection")
class mean_squared_error_with_hnm_for_one_class_detection_Prop(mx.operator.CustomOpProp):
    def __init__(self, hnm_ratio=10):
        super(mean_squared_error_with_hnm_for_one_class_detection_Prop, self).__init__(need_top_grad=False)
        self.hnm_ratio = hnm_ratio

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
        return mean_squared_error_with_hnm_for_one_class_detection(self.hnm_ratio)
