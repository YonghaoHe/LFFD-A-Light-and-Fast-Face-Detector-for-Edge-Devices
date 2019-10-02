# -*- coding: utf-8 -*-

import sys
import datetime
import os
import math
import logging
from ChasingTrainFramework_GeneralOneClassDetection import logging_GOCD
from ChasingTrainFramework_GeneralOneClassDetection import train_GOCD


# add mxnet python path to path env if need
mxnet_python_path = '/home/heyonghao/libs/incubator-mxnet/python'
sys.path.append(mxnet_python_path)
import mxnet

'''
init logging
'''
param_log_mode = 'w'
param_log_file_path = '../log/%s_%s.log' % (os.path.basename(__file__)[:-3], datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))  # 构建日志文件路径

'''
    data setting
'''
# pick file path for train set
param_trainset_pickle_file_path = os.path.join(os.path.dirname(__file__), '../data_provider_farm/data_folder/data_list_CCPD_train_debug.pkl')
# pick file path for val set
param_valset_pickle_file_path = ''

'''
    training setting
'''

# batchsize for training
param_train_batch_size = 32

# the ratio of neg image in a batch
param_neg_image_ratio = 0.1

# GPU index for training (single machine multi GPU)
param_GPU_idx_list = [0]

# input height for network
param_net_input_height = 640

# input width for network
param_net_input_width = 640

# the number of train loops
param_num_train_loops = 1000000

# the number of threads used for train dataiter
param_num_thread_train_dataiter = 6

# the number of threads used for val dataiter
param_num_thread_val_dataiter = 2

# training start index
param_start_index = 0

# the evaluation frequency for current model
param_validation_interval = 10000

# batchsize for validation
param_val_batch_size = 20

# the number of loops for each evaluation
param_num_val_loops = 0

# the path of pre-trained model
param_pretrained_model_param_path = ''

# the frequency of display, namely displaying every param_display_interval loops
param_display_interval = 20

# the frequency of metric update, less updates will boost the training speed (should less than param_display_interval)
param_train_metric_update_frequency = 5

# set save prefix (auto)
param_save_prefix = '../saved_model/' + os.path.basename(__file__)[:-3] + '_' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + \
                    '/' + os.path.basename(__file__)[:-3].replace('configuration', 'train')

# the frequency of model saving, namely saving the model params every param_model_save_interval loops
param_model_save_interval = 50000

# hard nagative mining ratio, needed by loss layer
param_hnm_ratio = 5

# init learning rate
param_learning_rate = 0.1
# weight decay
param_weight_decay = 0.00001
# momentum
param_momentum = 0.9

# learning rate scheduler -- MultiFactorScheduler
scheduler_step_list = [300000, 600000, ]
# multiply factor of scheduler
scheduler_factor = 0.1

# construct the learning rate scheduler
param_lr_scheduler = mxnet.lr_scheduler.MultiFactorScheduler(step=scheduler_step_list, factor=scheduler_factor)
# param_optimizer_name = 'adam'
# param_optimizer_params = {'learning_rate': param_learning_rate,
#                           'wd': param_weight_decay,
#                           'lr_scheduler': param_lr_scheduler,
#                           'begin_num_update': param_start_index}
param_optimizer_name = 'sgd'
param_optimizer_params = {'learning_rate': param_learning_rate,
                          'wd': param_weight_decay,
                          'lr_scheduler': param_lr_scheduler,
                          'momentum': param_momentum,
                          'begin_num_update': param_start_index}
'''
    data augmentation
'''

# trigger for horizon flip
param_enable_horizon_flip = True

# trigger for vertical flip
param_enable_vertical_flip = True

# trigger for brightness
param_enable_random_brightness = True
param_brightness_factors = {'min_factor': 0.5, 'max_factor': 1.5}

# trigger for saturation
param_enable_random_saturation = True
param_saturation_factors = {'min_factor': 0.5, 'max_factor': 1.5}

# trigger for contrast
param_enable_random_contrast = True
param_contrast_factors = {'min_factor': 0.5, 'max_factor': 1.5}

# trigger for blur
param_enable_blur = False
param_blur_factors = {'mode': 'random', 'sigma': 1}
param_blur_kernel_size_list = [3]

# negative image resize interval
param_neg_image_resize_factor_interval = [0.5, 3.5]

'''
    algorithm
'''
# the number of image channels
param_num_image_channel = 3

# the number of output scales (loss branches)
param_num_output_scales = 3

# feature map size for each scale
param_feature_map_size_list = [39, 19, 9]

# bbox lower bound for each scale
param_bbox_small_list = [64, 128, 256]
assert len(param_bbox_small_list) == param_num_output_scales
# bbox upper bound for each scale
param_bbox_large_list = [128, 256, 512]
assert len(param_bbox_large_list) == param_num_output_scales

# bbox gray lower bound for each scale
param_bbox_small_gray_list = [math.floor(v * 0.9) for v in param_bbox_small_list]
# bbox gray upper bound for each scale
param_bbox_large_gray_list = [math.ceil(v * 1.1) for v in param_bbox_large_list]

# the RF size of each scale used for normalization, here we use param_bbox_large_list for better regression
param_receptive_field_list = param_bbox_large_list
# RF stride for each scale
param_receptive_field_stride = [16, 32, 64]
# the start location of the first RF of each scale
param_receptive_field_center_start = [15, 31, 63]

# the sum of the number of output channels, 2 channels for classification and 4 for bbox regression
param_num_output_channels = 6

# -------------------------------------------------------------------------------------------
# print all params
orig_param_dict = vars()
param_names = [name for name in orig_param_dict.keys() if name.startswith('param_')]
param_dict = dict()
for name in param_names:
    param_dict[name] = orig_param_dict[name]


def run():
    logging_GOCD.init_logging(log_file_path=param_log_file_path,
                              log_file_mode=param_log_mode)

    logging.info('Preparing before training.')
    sys.path.append('..')
    from symbol_farm import symbol_64_512_16L_3scales_v1 as net

    net_symbol, data_names, label_names = net.get_net_symbol()
    net_initializer = mxnet.initializer.Xavier()

    logging.info('Get net symbol successfully.')

    # -----------------------------------------------------------------------------------------------
    # init dataiter
    from data_provider_farm.pickle_provider import PickleProvider
    from data_iterator_farm.multithread_dataiter_for_cross_entropy_v1 import Multithread_DataIter_for_CrossEntropy as DataIter

    train_data_provider = PickleProvider(param_trainset_pickle_file_path)
    train_dataiter = DataIter(
        mxnet_module=mxnet,
        num_threads=param_num_thread_train_dataiter,
        data_provider=train_data_provider,
        batch_size=param_train_batch_size,
        enable_horizon_flip=param_enable_horizon_flip,
        enable_vertical_flip=param_enable_vertical_flip,
        enable_random_brightness=param_enable_random_brightness,
        brightness_params=param_brightness_factors,
        enable_random_saturation=param_enable_random_saturation,
        saturation_params=param_saturation_factors,
        enable_random_contrast=param_enable_random_contrast,
        contrast_params=param_contrast_factors,
        enable_blur=param_enable_blur,
        blur_params=param_blur_factors,
        blur_kernel_size_list=param_blur_kernel_size_list,
        neg_image_ratio=param_neg_image_ratio,
        num_image_channels=param_num_image_channel,
        net_input_height=param_net_input_height,
        net_input_width=param_net_input_width,
        num_output_scales=param_num_output_scales,
        receptive_field_list=param_receptive_field_list,
        receptive_field_stride=param_receptive_field_stride,
        feature_map_size_list=param_feature_map_size_list,
        receptive_field_center_start=param_receptive_field_center_start,
        bbox_small_list=param_bbox_small_list,
        bbox_large_list=param_bbox_large_list,
        bbox_small_gray_list=param_bbox_small_gray_list,
        bbox_large_gray_list=param_bbox_large_gray_list,
        num_output_channels=param_num_output_channels,
        neg_image_resize_factor_interval=param_neg_image_resize_factor_interval
    )

    val_dataiter = None
    if param_valset_pickle_file_path != '' and param_val_batch_size != 0 and param_num_val_loops != 0 and param_num_thread_val_dataiter != 0:
        val_data_provider = PickleProvider(param_valset_pickle_file_path)
        val_dataiter = DataIter(
            mxnet_module=mxnet,
            num_threads=param_num_thread_val_dataiter,
            data_provider=val_data_provider,
            batch_size=param_val_batch_size,
            enable_horizon_flip=param_enable_horizon_flip,
            enable_vertical_flip=param_enable_vertical_flip,
            enable_random_brightness=param_enable_random_brightness,
            brightness_params=param_brightness_factors,
            enable_random_saturation=param_enable_random_saturation,
            saturation_params=param_saturation_factors,
            enable_random_contrast=param_enable_random_contrast,
            contrast_params=param_contrast_factors,
            enable_blur=param_enable_blur,
            blur_params=param_blur_factors,
            blur_kernel_size_list=param_blur_kernel_size_list,
            neg_image_ratio=param_neg_image_ratio,
            num_image_channels=param_num_image_channel,
            net_input_height=param_net_input_height,
            net_input_width=param_net_input_width,
            num_output_scales=param_num_output_scales,
            receptive_field_list=param_receptive_field_list,
            receptive_field_stride=param_receptive_field_stride,
            feature_map_size_list=param_feature_map_size_list,
            receptive_field_center_start=param_receptive_field_center_start,
            bbox_small_list=param_bbox_small_list,
            bbox_large_list=param_bbox_large_list,
            bbox_small_gray_list=param_bbox_small_gray_list,
            bbox_large_gray_list=param_bbox_large_gray_list,
            num_output_channels=param_num_output_channels,
            neg_image_resize_factor_interval=param_neg_image_resize_factor_interval

        )
    # ---------------------------------------------------------------------------------------------
    # init metric
    from metric_farm.metric_default import Metric

    train_metric = Metric(param_num_output_scales)
    val_metric = None
    if val_dataiter is not None:
        val_metric = Metric(param_num_output_scales)

    train_GOCD.start_train(
        param_dict=param_dict,
        mxnet_module=mxnet,
        context=[mxnet.gpu(i) for i in param_GPU_idx_list],
        train_dataiter=train_dataiter,
        train_metric=train_metric,
        train_metric_update_frequency=param_train_metric_update_frequency,
        num_train_loops=param_num_train_loops,
        val_dataiter=val_dataiter,
        val_metric=val_metric,
        num_val_loops=param_num_val_loops,
        validation_interval=param_validation_interval,
        optimizer_name=param_optimizer_name,
        optimizer_params=param_optimizer_params,
        net_symbol=net_symbol,
        net_initializer=net_initializer,
        net_data_names=data_names,
        net_label_names=label_names,
        pretrained_model_param_path=param_pretrained_model_param_path,
        display_interval=param_display_interval,
        save_prefix=param_save_prefix,
        model_save_interval=param_model_save_interval,
        start_index=param_start_index)


if __name__ == '__main__':
    run()
