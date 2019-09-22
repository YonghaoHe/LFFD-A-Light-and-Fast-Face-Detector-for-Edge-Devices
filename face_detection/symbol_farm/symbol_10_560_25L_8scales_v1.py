# -*- coding: utf-8 -*-
"""
"""
import os
import sys
sys.path.append('..')
from config_farm import configuration_10_560_25L_8scales_v1 as cfg
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
                   'mask_4', 'label_4',
                   'mask_5', 'label_5',
                   'mask_6', 'label_6',
                   'mask_7', 'label_7',
                   'mask_8', 'label_8', ]

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
    label_5 = mxnet.symbol.Variable(name='label_5', shape=(cfg.param_train_batch_size, cfg.param_num_output_channels, cfg.param_feature_map_size_list[4], cfg.param_feature_map_size_list[4]))
    mask_5 = mxnet.symbol.Variable(name='mask_5', shape=(cfg.param_train_batch_size, cfg.param_num_output_channels, cfg.param_feature_map_size_list[4], cfg.param_feature_map_size_list[4]))
    label_6 = mxnet.symbol.Variable(name='label_6', shape=(cfg.param_train_batch_size, cfg.param_num_output_channels, cfg.param_feature_map_size_list[5], cfg.param_feature_map_size_list[5]))
    mask_6 = mxnet.symbol.Variable(name='mask_6', shape=(cfg.param_train_batch_size, cfg.param_num_output_channels, cfg.param_feature_map_size_list[5], cfg.param_feature_map_size_list[5]))
    label_7 = mxnet.symbol.Variable(name='label_7', shape=(cfg.param_train_batch_size, cfg.param_num_output_channels, cfg.param_feature_map_size_list[6], cfg.param_feature_map_size_list[6]))
    mask_7 = mxnet.symbol.Variable(name='mask_7', shape=(cfg.param_train_batch_size, cfg.param_num_output_channels, cfg.param_feature_map_size_list[6], cfg.param_feature_map_size_list[6]))
    label_8 = mxnet.symbol.Variable(name='label_8', shape=(cfg.param_train_batch_size, cfg.param_num_output_channels, cfg.param_feature_map_size_list[7], cfg.param_feature_map_size_list[7]))
    mask_8 = mxnet.symbol.Variable(name='mask_8', shape=(cfg.param_train_batch_size, cfg.param_num_output_channels, cfg.param_feature_map_size_list[7], cfg.param_feature_map_size_list[7]))

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
    # for scale [10,15]
    if deploy_flag:
        predict_score_1, predict_bbox_1 = loss_branch(relu8, 'conv8', deploy_flag=deploy_flag)
    else:
        loss_score_1, loss_bbox_1 = loss_branch(relu8, 'conv8', mask=mask_1, label=label_1)

    # conv block 9 ----------------------------------------------------------------------------------------
    conv9 = mxnet.symbol.Convolution(data=relu8,
                                     kernel=(3, 3),
                                     stride=(1, 1),
                                     pad=(1, 1),
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

    # Residual 4:
    conv10 = conv8 + conv10
    relu10 = mxnet.symbol.Activation(data=conv10, act_type='relu', name='relu_conv10')

    # loss 2 RF:71 ----------------------------------------------------------------------------------------------------
    # for scale [15,20]
    if deploy_flag:
        predict_score_2, predict_bbox_2 = loss_branch(relu10, 'conv10', deploy_flag=deploy_flag)
    else:
        loss_score_2, loss_bbox_2 = loss_branch(relu10, 'conv10', mask=mask_2, label=label_2)

    # conv block 11 ----------------------------------------------------------------------------------------
    conv11 = mxnet.symbol.Convolution(data=relu10,
                                      kernel=(3, 3),
                                      stride=(2, 2),
                                      pad=(0, 0),
                                      num_filter=num_filters_list[1],
                                      name='conv11')

    relu11 = mxnet.symbol.Activation(data=conv11, act_type='relu', name='relu_conv11')

    # conv block 12 ----------------------------------------------------------------------------------------
    conv12 = mxnet.symbol.Convolution(data=relu11,
                                      kernel=(3, 3),
                                      stride=(1, 1),
                                      pad=(1, 1),
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

    # Residual
    conv13 = conv11 + conv13
    relu13 = mxnet.symbol.Activation(data=conv13, act_type='relu', name='relu_conv13')

    # loss 3 RF:111 ----------------------------------------------------------------------------------------------------
    # for scale [20,40]
    if deploy_flag:
        predict_score_3, predict_bbox_3 = loss_branch(relu13, 'conv13', deploy_flag=deploy_flag)
    else:
        loss_score_3, loss_bbox_3 = loss_branch(relu13, 'conv13', mask=mask_3, label=label_3)

    # conv block 14 ----------------------------------------------------------------------------------------
    conv14 = mxnet.symbol.Convolution(data=relu13,
                                      kernel=(3, 3),
                                      stride=(1, 1),
                                      pad=(1, 1),
                                      num_filter=num_filters_list[1],
                                      name='conv14')

    relu14 = mxnet.symbol.Activation(data=conv14, act_type='relu', name='relu_conv14')

    # conv block 15 ----------------------------------------------------------------------------------------
    conv15 = mxnet.symbol.Convolution(data=relu14,
                                      kernel=(3, 3),
                                      stride=(1, 1),
                                      pad=(1, 1),
                                      num_filter=num_filters_list[1],
                                      name='conv15')

    # Residual
    conv15 = conv13 + conv15
    relu15 = mxnet.symbol.Activation(data=conv15, act_type='relu', name='relu_conv15')

    # loss 4 RF:143 ----------------------------------------------------------------------------------------------------
    # for scale [40,70]
    if deploy_flag:
        predict_score_4, predict_bbox_4 = loss_branch(relu15, 'conv15', deploy_flag=deploy_flag)
    else:
        loss_score_4, loss_bbox_4 = loss_branch(relu15, 'conv15', mask=mask_4, label=label_4)

    # conv block 16 ----------------------------------------------------------------------------------------
    conv16 = mxnet.symbol.Convolution(data=relu15,
                                      kernel=(3, 3),
                                      stride=(2, 2),
                                      pad=(0, 0),
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

    relu17 = mxnet.symbol.Activation(data=conv17, act_type='relu', name='relu_conv17')

    # conv block 18 ----------------------------------------------------------------------------------------
    conv18 = mxnet.symbol.Convolution(data=relu17,
                                      kernel=(3, 3),
                                      stride=(1, 1),
                                      pad=(1, 1),
                                      num_filter=num_filters_list[2],
                                      name='conv18')

    # Residual
    conv18 = conv16 + conv18
    relu18 = mxnet.symbol.Activation(data=conv18, act_type='relu', name='relu_conv18')

    # loss 5 RF:223 ----------------------------------------------------------------------------------------------------
    # for scale [70,110]
    if deploy_flag:
        predict_score_5, predict_bbox_5 = loss_branch(relu18, 'conv18', deploy_flag=deploy_flag)
    else:
        loss_score_5, loss_bbox_5 = loss_branch(relu18, 'conv18', mask=mask_5, label=label_5)

    # conv block 19 ----------------------------------------------------------------------------------------
    conv19 = mxnet.symbol.Convolution(data=relu18,
                                      kernel=(3, 3),
                                      stride=(2, 2),
                                      pad=(0, 0),
                                      num_filter=num_filters_list[2],
                                      name='conv19')

    relu19 = mxnet.symbol.Activation(data=conv19, act_type='relu', name='relu_conv19')

    # conv block 20 ----------------------------------------------------------------------------------------
    conv20 = mxnet.symbol.Convolution(data=relu19,
                                      kernel=(3, 3),
                                      stride=(1, 1),
                                      pad=(1, 1),
                                      num_filter=num_filters_list[2],
                                      name='conv20')

    relu20 = mxnet.symbol.Activation(data=conv20, act_type='relu', name='relu_conv20')

    # conv block 21 ----------------------------------------------------------------------------------------
    conv21 = mxnet.symbol.Convolution(data=relu20,
                                      kernel=(3, 3),
                                      stride=(1, 1),
                                      pad=(1, 1),
                                      num_filter=num_filters_list[2],
                                      name='conv21')

    # Residual
    conv21 = conv19 + conv21
    relu21 = mxnet.symbol.Activation(data=conv21, act_type='relu', name='relu_conv21')

    # loss 6 RF:383 ----------------------------------------------------------------------------------------------------
    # for scale [110,190]
    if deploy_flag:
        predict_score_6, predict_bbox_6 = loss_branch(relu21, 'conv21', deploy_flag=deploy_flag)
    else:
        loss_score_6, loss_bbox_6 = loss_branch(relu21, 'conv21', mask=mask_6, label=label_6)

    # conv block 22 ----------------------------------------------------------------------------------------
    conv22 = mxnet.symbol.Convolution(data=relu21,
                                      kernel=(3, 3),
                                      stride=(1, 1),
                                      pad=(1, 1),
                                      num_filter=num_filters_list[2],
                                      name='conv22')

    relu22 = mxnet.symbol.Activation(data=conv22, act_type='relu', name='relu_conv22')

    # conv block 23 ----------------------------------------------------------------------------------------
    conv23 = mxnet.symbol.Convolution(data=relu22,
                                      kernel=(3, 3),
                                      stride=(1, 1),
                                      pad=(1, 1),
                                      num_filter=num_filters_list[2],
                                      name='conv23')

    # Residual
    conv23 = conv21 + conv23
    relu23 = mxnet.symbol.Activation(data=conv23, act_type='relu', name='relu_conv23')

    # loss 7 RF:511 ----------------------------------------------------------------------------------------------------
    # for scale [190,290]
    if deploy_flag:
        predict_score_7, predict_bbox_7 = loss_branch(relu23, 'conv23', deploy_flag=deploy_flag)
    else:
        loss_score_7, loss_bbox_7 = loss_branch(relu23, 'conv23', mask=mask_7, label=label_7)

    # conv block 24 ----------------------------------------------------------------------------------------
    conv24 = mxnet.symbol.Convolution(data=relu23,
                                      kernel=(3, 3),
                                      stride=(1, 1),
                                      pad=(1, 1),
                                      num_filter=num_filters_list[2],
                                      name='conv24')

    relu24 = mxnet.symbol.Activation(data=conv24, act_type='relu', name='relu_conv24')

    # conv block 25 ----------------------------------------------------------------------------------------
    conv25 = mxnet.symbol.Convolution(data=relu24,
                                      kernel=(3, 3),
                                      stride=(1, 1),
                                      pad=(1, 1),
                                      num_filter=num_filters_list[2],
                                      name='conv25')

    # Residual
    conv25 = conv23 + conv25
    relu25 = mxnet.symbol.Activation(data=conv25, act_type='relu', name='relu_conv25')

    # loss 8 RF:639 ----------------------------------------------------------------------------------------------------
    # for scale [290,410]
    if deploy_flag:
        predict_score_8, predict_bbox_8 = loss_branch(relu25, 'conv25', deploy_flag=deploy_flag)
    else:
        loss_score_8, loss_bbox_8 = loss_branch(relu25, 'conv25', mask=mask_8, label=label_8)

    if deploy_flag:
        net = mxnet.symbol.Group([predict_score_1, predict_bbox_1,
                                  predict_score_2, predict_bbox_2,
                                  predict_score_3, predict_bbox_3,
                                  predict_score_4, predict_bbox_4,
                                  predict_score_5, predict_bbox_5,
                                  predict_score_6, predict_bbox_6,
                                  predict_score_7, predict_bbox_7,
                                  predict_score_8, predict_bbox_8])

        return net
    else:
        net = mxnet.symbol.Group([loss_score_1, loss_bbox_1,
                                  loss_score_2, loss_bbox_2,
                                  loss_score_3, loss_bbox_3,
                                  loss_score_4, loss_bbox_4,
                                  loss_score_5, loss_bbox_5,
                                  loss_score_6, loss_bbox_6,
                                  loss_score_7, loss_bbox_7,
                                  loss_score_8, loss_bbox_8])

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
