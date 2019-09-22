import mxnet as mx


class cross_entropy_with_hnm_for_one_class_detection(mx.operator.CustomOp):

    def __init__(self, hnm_ratio):
        super(cross_entropy_with_hnm_for_one_class_detection, self).__init__()
        self.hnm_ratio = int(hnm_ratio)

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
        CE_gradient = pred_softmax - label  # Standard CE gradient
        loss_mask = mx.ndarray.ones((CE_gradient.shape[0], 1, CE_gradient.shape[2], CE_gradient.shape[3]), ctx=CE_gradient.context)

        if self.hnm_ratio > 0:
            pos_flag = (label[:, 0, :, :] > 0.5)
            pos_num = mx.ndarray.sum(pos_flag).asnumpy()[0]  # 得到正样本的个数

            if pos_num > 0:
                neg_flag = (label[:, 1, :, :] > 0.5)
                neg_num = mx.ndarray.sum(neg_flag).asnumpy()[0]
                neg_num_selected = min(int(self.hnm_ratio * pos_num), int(neg_num))
                neg_prob = pred_softmax[:, 1, :, :] * neg_flag  # non-negative value
                neg_prob_sort = mx.ndarray.sort(neg_prob.reshape((1, -1)), is_ascend=True)

                prob_threshold = neg_prob_sort[0][neg_num_selected].asnumpy()[0]
                neg_grad_flag = (neg_prob <= prob_threshold)
                loss_mask = mx.ndarray.logical_or(neg_grad_flag, pos_flag)
            else:
                neg_choice_ratio = 0.1
                neg_num_selected = int(pred_softmax[:, 1, :, :].size * neg_choice_ratio)
                neg_prob = pred_softmax[:, 1, :, :]
                neg_prob_sort = mx.ndarray.sort(neg_prob.reshape((1, -1)), is_ascend=True)
                prob_threshold = neg_prob_sort[0][neg_num_selected].asnumpy()[0]
                loss_mask = (neg_prob <= prob_threshold)

            for i in range(CE_gradient.shape[1]):
                CE_gradient[:, i, :, :] *= loss_mask * mask[:, i, :, :]

        CE_gradient /= mx.ndarray.sum(loss_mask).asnumpy()[0]

        self.assign(in_grad[0], req[0], CE_gradient)


@mx.operator.register("cross_entropy_with_hnm_for_one_class_detection")
class cross_entropy_with_hnm_for_one_class_detection_Prop(mx.operator.CustomOpProp):
    def __init__(self, hnm_ratio=5):
        super(cross_entropy_with_hnm_for_one_class_detection_Prop, self).__init__(need_top_grad=False)
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
        return cross_entropy_with_hnm_for_one_class_detection(self.hnm_ratio)
