# -*- coding: utf-8 -*-
"""
network description:
network structure:
"""
import os
import sys
sys.path.append('..')
from config_farm import configuration_10_160_17L_4scales_v1 as cfg
from ChasingTrainFramework_GeneralOneClassDetection.loss_layer_farm.cross_entropy_with_hnm_for_one_class_detection import *
import mxnet

num_filters_list = [32, 64, 128, 256]


def loss_branch(input_data, prefix_name, mask=None, label=None, deploy_flag=False):
    branch_conv1 = mxnet.symbol.Convolution(data=input_data,
                                            kernel=(1, 1),
                                            stride=(1, 1),
                                            pad=(0, 0),
                                            num_filter=num_filters_list[2],
                                            name=prefix_name + '_1')
    branch_relu1 = mxnet.symbol.Activation(data=branch_conv1, act_type='relu', name='relu_' + prefix_name + '_1')

    branch_conv2_score = mxnet.symbol.Convolution(data=branch_relu1,
                                                  kernel=(1, 1),
                                                  stride=(1, 1),
                                                  pad=(0, 0),
                                                  num_filter=num_filters_list[2],
                                                  name=prefix_name + '_2_score')
    branch_relu2_score = mxnet.symbol.Activation(data=branch_conv2_score, act_type='relu', name='relu_' + prefix_name + '_2_score')

    branch_conv3_score = mxnet.symbol.Convolution(data=branch_relu2_score,
                                                  kernel=(1, 1),
                                                  stride=(1, 1),
                                                  pad=(0, 0),
                                                  num_filter=2,
                                                  name=prefix_name + '_3_score')

    branch_conv2_bbox = mxnet.symbol.Convolution(data=branch_relu1,
                                                 kernel=(1, 1),
                                                 stride=(1, 1),
                                                 pad=(0, 0),
                                                 num_filter=num_filters_list[2],
                                                 name=prefix_name + '_2_bbox')
    branch_relu2_bbox = mxnet.symbol.Activation(data=branch_conv2_bbox, act_type='relu', name='relu_' + prefix_name + '_2_bbox')

    branch_conv3_bbox = mxnet.symbol.Convolution(data=branch_relu2_bbox,
                                                 kernel=(1, 1),
                                                 stride=(1, 1),
                                                 pad=(0, 0),
                                                 num_filter=4,
                                                 name=prefix_name + '_3_bbox')

    if deploy_flag:
        predict_score = mxnet.symbol.softmax(data=branch_conv3_score, axis=1)
        predict_score = mxnet.symbol.slice_axis(predict_score, axis=1, begin=0, end=1)

        predict_bbox = branch_conv3_bbox

        return predict_score, predict_bbox
    else:

        mask_score = mxnet.symbol.slice_axis(mask, axis=1, begin=0, end=2)
        label_score = mxnet.symbol.slice_axis(label, axis=1, begin=0, end=2)
        loss_score = mxnet.symbol.Custom(pred=branch_conv3_score, label=label_score, mask=mask_score, hnm_ratio=cfg.param_hnm_ratio,
                                         op_type='cross_entropy_with_hnm_for_one_class_detection', name=prefix_name + '_loss_score')

        mask_bbox = mxnet.symbol.slice_axis(mask, axis=1, begin=2, end=6)
        predict_bbox = branch_conv3_bbox * mask_bbox
        label_bbox = mxnet.symbol.slice_axis(label, axis=1, begin=2, end=6) * mask_bbox
        loss_bbox = mxnet.symbol.LinearRegressionOutput(data=predict_bbox, label=label_bbox, name=prefix_name + '_loss_bbox')

        return loss_score, loss_bbox


def get_net_symbol(deploy_flag=False):
    data_names = ['data']

    label_names = ['mask_1', 'label_1',
                   'mask_2', 'label_2',
                   'mask_3', 'label_3',
                   'mask_4', 'label_4',]

    # batch data
    data = mxnet.symbol.Variable(name='data', shape=(cfg.param_train_batch_size, cfg.param_num_image_channel, cfg.param_net_input_height, cfg.param_net_input_width))
    label_1 = mxnet.symbol.Variable(name='label_1', shape=(cfg.param_train_batch_size, cfg.param_num_output_channels, cfg.param_feature_map_size_list[0], cfg.param_feature_map_size_list[0]))
    mask_1 = mxnet.symbol.Variable(name='mask_1', shape=(cfg.param_train_batch_size, cfg.param_num_output_channels, cfg.param_feature_map_size_list[0], cfg.param_feature_map_size_list[0]))
    label_2 = mxnet.symbol.Variable(name='label_2', shape=(cfg.param_train_batch_size, cfg.param_num_output_channels, cfg.param_feature_map_size_list[1], cfg.param_feature_map_size_list[1]))
    mask_2 = mxnet.symbol.Variable(name='mask_2', shape=(cfg.param_train_batch_size, cfg.param_num_output_channels, cfg.param_feature_map_size_list[1], cfg.param_feature_map_size_list[1]))
    label_3 = mxnet.symbol.Variable(name='label_3', shape=(cfg.param_train_batch_size, cfg.param_num_output_channels, cfg.param_feature_map_size_list[2], cfg.param_feature_map_size_list[2]))
    mask_3 = mxnet.symbol.Variable(name='mask_3', shape=(cfg.param_train_batch_size, cfg.param_num_output_channels, cfg.param_feature_map_size_list[2], cfg.param_feature_map_size_list[2]))
    label_4 = mxnet.symbol.Variable(name='label_4', shape=(cfg.param_train_batch_size, cfg.param_num_output_channels, cfg.param_feature_map_size_list[3], cfg.param_feature_map_size_list[3]))
    mask_4 = mxnet.symbol.Variable(name='mask_4', shape=(cfg.param_train_batch_size, cfg.param_num_output_channels, cfg.param_feature_map_size_list[3], cfg.param_feature_map_size_list[3]))

    data = (data - 127.5) / 127.5

    # conv block 1 ---------------------------------------------------------------------------------------
    conv1 = mxnet.symbol.Convolution(data=data,
                                     kernel=(3, 3),
                                     stride=(2, 2),
                                     pad=(0, 0),
                                     num_filter=num_filters_list[1],
                                     name='conv1')
    relu1 = mxnet.symbol.Activation(data=conv1, act_type='relu', name='relu_conv1')

    # conv block 2 ----------------------------------------------------------------------------------------
    conv2 = mxnet.symbol.Convolution(data=relu1,
                                     kernel=(3, 3),
                                     stride=(2, 2),
                                     pad=(0, 0),
                                     num_filter=num_filters_list[1],
                                     name='conv2')
    relu2 = mxnet.symbol.Activation(data=conv2, act_type='relu', name='relu_conv2')

    # conv block 3 ----------------------------------------------------------------------------------------
    conv3 = mxnet.symbol.Convolution(data=relu2,
                                     kernel=(3, 3),
                                     stride=(1, 1),
                                     pad=(1, 1),
                                     num_filter=num_filters_list[1],
                                     name='conv3')
    relu3 = mxnet.symbol.Activation(data=conv3, act_type='relu', name='relu_conv3')

    # conv block 4 ----------------------------------------------------------------------------------------
    conv4 = mxnet.symbol.Convolution(data=relu3,
                                     kernel=(3, 3),
                                     stride=(1, 1),
                                     pad=(1, 1),
                                     num_filter=num_filters_list[1],
                                     name='conv4')

    # Residual 1:
    conv4 = conv2 + conv4
    relu4 = mxnet.symbol.Activation(data=conv4, act_type='relu', name='relu_conv4')

    # conv block 5 ----------------------------------------------------------------------------------------
    conv5 = mxnet.symbol.Convolution(data=relu4,
                                     kernel=(3, 3),
                                     stride=(1, 1),
                                     pad=(1, 1),
                                     num_filter=num_filters_list[1],
                                     name='conv5')
    relu5 = mxnet.symbol.Activation(data=conv5, act_type='relu', name='relu_conv5')

    # conv block 6 ----------------------------------------------------------------------------------------
    conv6 = mxnet.symbol.Convolution(data=relu5,
                                     kernel=(3, 3),
                                     stride=(1, 1),
                                     pad=(1, 1),
                                     num_filter=num_filters_list[1],
                                     name='conv6')

    # Residual 2:
    conv6 = conv4 + conv6
    relu6 = mxnet.symbol.Activation(data=conv6, act_type='relu', name='relu_conv6')

    # conv block 7 ----------------------------------------------------------------------------------------
    conv7 = mxnet.symbol.Convolution(data=relu6,
                                     kernel=(3, 3),
                                     stride=(1, 1),
                                     pad=(1, 1),
                                     num_filter=num_filters_list[1],
                                     name='conv7')

    relu7 = mxnet.symbol.Activation(data=conv7, act_type='relu', name='relu_conv7')

    # conv block 8 ----------------------------------------------------------------------------------------
    conv8 = mxnet.symbol.Convolution(data=relu7,
                                     kernel=(3, 3),
                                     stride=(1, 1),
                                     pad=(1, 1),
                                     num_filter=num_filters_list[1],
                                     name='conv8')

    # Residual 3:
    conv8 = conv6 + conv8
    relu8 = mxnet.symbol.Activation(data=conv8, act_type='relu', name='relu_conv8')

    # loss 1 RF:55 ----------------------------------------------------------------------------------------------------
    # for scale [10,20]----------------------------------------------------------------------------------------
    if deploy_flag:
        predict_score_1, predict_bbox_1 = loss_branch(relu8, 'conv8', deploy_flag=deploy_flag)
    else:
        loss_score_1, loss_bbox_1 = loss_branch(relu8, 'conv8', mask=mask_1, label=label_1)

    # conv block 9 ----------------------------------------------------------------------------------------
    conv9 = mxnet.symbol.Convolution(data=relu8,
                                     kernel=(3, 3),
                                     stride=(2, 2),
                                     pad=(0, 0),
                                     num_filter=num_filters_list[1],
                                     name='conv9')

    relu9 = mxnet.symbol.Activation(data=conv9, act_type='relu', name='relu_conv9')

    # conv block 10 ----------------------------------------------------------------------------------------
    conv10 = mxnet.symbol.Convolution(data=relu9,
                                      kernel=(3, 3),
                                      stride=(1, 1),
                                      pad=(1, 1),
                                      num_filter=num_filters_list[1],
                                      name='conv10')
    relu10 = mxnet.symbol.Activation(data=conv10, act_type='relu', name='relu_conv10')

    # conv block 11 ----------------------------------------------------------------------------------------
    conv11 = mxnet.symbol.Convolution(data=relu10,
                                      kernel=(3, 3),
                                      stride=(1, 1),
                                      pad=(1, 1),
                                      num_filter=num_filters_list[1],
                                      name='conv11')
    # Residual
    conv11 = conv9 + conv11
    relu11 = mxnet.symbol.Activation(data=conv11, act_type='relu', name='relu_conv11')

    # loss 2 RF:95 ----------------------------------------------------------------------------------------------------
    # for scale [20,40]----------------------------------------------------------------------------------------
    if deploy_flag:
        predict_score_2, predict_bbox_2 = loss_branch(relu11, 'conv11', deploy_flag=deploy_flag)
    else:
        loss_score_2, loss_bbox_2 = loss_branch(relu11, 'conv11', mask=mask_2, label=label_2)

    # conv block 12 ----------------------------------------------------------------------------------------
    conv12 = mxnet.symbol.Convolution(data=relu11,
                                      kernel=(3, 3),
                                      stride=(2, 2),
                                      pad=(0, 0),
                                      num_filter=num_filters_list[1],
                                      name='conv12')
    relu12 = mxnet.symbol.Activation(data=conv12, act_type='relu', name='relu_conv12')

    # conv block 13 ----------------------------------------------------------------------------------------
    conv13 = mxnet.symbol.Convolution(data=relu12,
                                      kernel=(3, 3),
                                      stride=(1, 1),
                                      pad=(1, 1),
                                      num_filter=num_filters_list[1],
                                      name='conv13')
    relu13 = mxnet.symbol.Activation(data=conv13, act_type='relu', name='relu_conv13')

    # conv block 14 ----------------------------------------------------------------------------------------
    conv14 = mxnet.symbol.Convolution(data=relu13,
                                      kernel=(3, 3),
                                      stride=(1, 1),
                                      pad=(1, 1),
                                      num_filter=num_filters_list[1],
                                      name='conv14')
    # Residual
    conv14 = conv12 + conv14
    relu14 = mxnet.symbol.Activation(data=conv14, act_type='relu', name='relu_conv14')

    # loss 3 RF:175 ----------------------------------------------------------------------------------------------------
    # for scale [40,80]----------------------------------------------------------------------------------------
    if deploy_flag:
        predict_score_3, predict_bbox_3 = loss_branch(relu14, 'conv14', deploy_flag=deploy_flag)
    else:
        loss_score_3, loss_bbox_3 = loss_branch(relu14, 'conv14', mask=mask_3, label=label_3)

    # conv block 15 ----------------------------------------------------------------------------------------
    conv15 = mxnet.symbol.Convolution(data=relu14,
                                      kernel=(3, 3),
                                      stride=(2, 2),
                                      pad=(0, 0),
                                      num_filter=num_filters_list[2],
                                      name='conv15')
    relu15 = mxnet.symbol.Activation(data=conv15, act_type='relu', name='relu_conv15')

    # conv block 16 ----------------------------------------------------------------------------------------
    conv16 = mxnet.symbol.Convolution(data=relu15,
                                      kernel=(3, 3),
                                      stride=(1, 1),
                                      pad=(1, 1),
                                      num_filter=num_filters_list[2],
                                      name='conv16')
    relu16 = mxnet.symbol.Activation(data=conv16, act_type='relu', name='relu_conv16')

    # conv block 17 ----------------------------------------------------------------------------------------
    conv17 = mxnet.symbol.Convolution(data=relu16,
                                      kernel=(3, 3),
                                      stride=(1, 1),
                                      pad=(1, 1),
                                      num_filter=num_filters_list[2],
                                      name='conv17')
    # Residual
    conv17 = conv15 + conv17
    relu17 = mxnet.symbol.Activation(data=conv17, act_type='relu', name='relu_conv17')

    # loss 4 RF:335 ----------------------------------------------------------------------------------------------------
    # for scale [80, 160]----------------------------------------------------------------------------------------
    if deploy_flag:
        predict_score_4, predict_bbox_4 = loss_branch(relu17, 'conv17', deploy_flag=deploy_flag)
    else:
        loss_score_4, loss_bbox_4 = loss_branch(relu17, 'conv17', mask=mask_4, label=label_4)

    if deploy_flag:
        net = mxnet.symbol.Group([predict_score_1, predict_bbox_1,
                                  predict_score_2, predict_bbox_2,
                                  predict_score_3, predict_bbox_3,
                                  predict_score_4, predict_bbox_4])

        return net
    else:

        net = mxnet.symbol.Group([loss_score_1, loss_bbox_1,
                                  loss_score_2, loss_bbox_2,
                                  loss_score_3, loss_bbox_3,
                                  loss_score_4, loss_bbox_4])

        return net, data_names, label_names


def run_get_net_symbol_for_train():
    my_symbol, _, __ = get_net_symbol()

    shape = {'data': (cfg.param_train_batch_size, cfg.param_num_image_channel, cfg.param_net_input_height, cfg.param_net_input_width)}
    print(mxnet.viz.print_summary(my_symbol, shape=shape))
    arg_names = my_symbol.list_arguments()
    aux_names = my_symbol.list_auxiliary_states()
    arg_shapes, out_shapes, _ = my_symbol.infer_shape()
    print(arg_names)
    print(aux_names)
    print(my_symbol.list_outputs())
    print(out_shapes)


if __name__ == '__main__':
    run_get_net_symbol_for_train()
    deploy_net = get_net_symbol(deploy_flag=True)
    deploy_net.save(os.path.basename(__file__).replace('.py', '_deploy.json'))
