# -*- coding: utf-8 -*-
import os
import logging
import time


class Solver(object):

    def __init__(self,
                 mxnet_module,
                 trainset_dataiter,
                 net_symbol,
                 net_initializer,
                 optimizer_name,
                 optimizer_params,
                 data_names,
                 label_names,
                 context,
                 num_train_loops,
                 train_metric,
                 display_interval=10,
                 val_evaluation_interval=100,
                 valset_dataiter=None,
                 val_metric=None,
                 num_val_loops=0,
                 pretrained_model_param_path=None,
                 save_prefix=None,
                 start_index=0,
                 model_save_interval=None,
                 train_metric_update_frequency=1):
        self.mxnet_module = mxnet_module
        self.trainset_dataiter = trainset_dataiter
        self.valset_dataiter = valset_dataiter
        self.net_symbol = net_symbol
        self.net_initializer = net_initializer
        self.data_names = data_names
        self.label_names = label_names
        self.input_names = data_names + label_names
        self.context = context
        self.optimizer_name = optimizer_name
        self.optimizer_params = optimizer_params
        self.num_train_loops = num_train_loops
        self.num_val_loops = num_val_loops
        self.train_metric = train_metric
        self.val_metric = val_metric
        self.display_interval = display_interval
        self.val_evaluation_interval = val_evaluation_interval
        self.save_prefix = save_prefix
        self.start_index = start_index
        self.pretrained_model_param_path = pretrained_model_param_path
        self.model_save_interval = model_save_interval

        self.train_metric_update_frequency = \
            train_metric_update_frequency if train_metric_update_frequency <= display_interval else display_interval

        self.module = self.mxnet_module.module.Module(symbol=self.net_symbol,
                                                      data_names=self.data_names,
                                                      label_names=self.label_names,
                                                      context=self.context)

    def __init_module(self):
        arg_names = self.net_symbol.list_arguments()
        arg_shapes, _, __ = self.net_symbol.infer_shape()
        data_name_shape = [x for x in zip(arg_names, arg_shapes) if x[0] in self.data_names]
        label_name_shape_temp = [x for x in zip(arg_names, arg_shapes) if x[0] in self.label_names]
        # rearrange  according to label_names
        label_name_shape = []
        for label_name in self.label_names:
            for temp_item in label_name_shape_temp:
                if temp_item[0] == label_name:
                    label_name_shape.append(temp_item)
                    break

        self.module.bind(data_shapes=data_name_shape,
                         label_shapes=label_name_shape,
                         for_training=True,
                         grad_req='write')

        if self.pretrained_model_param_path:
            self.load_checkpoint()
            self.module.params_initialized = True
        else:
            self.module.init_params(initializer=self.net_initializer,
                                    allow_missing=True)
        self.module.init_optimizer(kvstore='device',
                                   optimizer=self.optimizer_name,
                                   optimizer_params=self.optimizer_params)

    def fit(self):
        self.__init_module()

        logging.info('Start training in %s.--------------------------------------------', str(self.context))
        sum_time = 0
        for i in range(self.start_index + 1, self.num_train_loops + 1):
            start = time.time()
            batch = self.trainset_dataiter.next()

            self.module.forward(data_batch=batch, is_train=True)
            self.module.backward()

            # update parameters----------------------------------------------------------------------------------------
            self.module.update()
            outputs = [output for output in self.module.get_outputs() if not output.wait_to_read()]

            # display training process----------------------------------------------------------------------------------
            if i % self.train_metric_update_frequency == 0:
                self.train_metric.update(batch.label, outputs)

            sum_time += (time.time() - start)

            if i % self.display_interval == 0:

                names, values = self.train_metric.get()

                logging.info('Iter[%d] -- Time elapsed: %.1f s. Speed: %.1f images/s.',
                             i, sum_time, self.display_interval * self.trainset_dataiter.get_batch_size() / sum_time)
                for name, value in zip(names, values):
                    logging.info('%s: --> %.4f', name, value)

                self.train_metric.reset()
                sum_time = 0

            if i % self.val_evaluation_interval == 0 and self.num_val_loops:

                logging.info('Start validating-------------------------------------------')
                for val_loop in range(self.num_val_loops):
                    val_batch = self.valset_dataiter.next()

                    self.module.forward(data_batch=val_batch, is_train=False)
                    outputs = [output for output in self.module.get_outputs() if not output.wait_to_read()]
                    self.val_metric.update(val_batch.label, outputs)
                names, values = self.val_metric.get()
                logging.info('Iter[%d] validation metric ------------- ', i)
                for name, value in zip(names, values):
                    logging.info('%s: --> %.4f', name, value)
                logging.info('End validating ---------------------------------------------')
                self.val_metric.reset()

            # save checkpoint--------------------------------------------------------------------------------
            if i % self.model_save_interval == 0:
                self.save_checkpoint(i)

    def save_checkpoint(self, loop):
        logging.info('\n<---------- Save checkpoint---------->')
        save_model_name = '%s_iter_%d.params' % (self.save_prefix, loop)
        if not os.path.exists(os.path.dirname(save_model_name)):
            os.makedirs(os.path.dirname(save_model_name))
        temp_arg_name_arrays, temp_aux_name_arrays = self.module.get_params()
        # save model params
        save_dict = {('arg:%s' % k): v.as_in_context(self.mxnet_module.cpu()) for k, v in temp_arg_name_arrays.items() if k not in self.input_names}
        save_dict.update({('aux:%s' % k): v.as_in_context(self.mxnet_module.cpu()) for k, v in temp_aux_name_arrays.items()})
        self.mxnet_module.nd.save(save_model_name, save_dict)
        logging.info('Iter[%d] <--Save params to file: %s-->', loop, save_model_name)

    def load_checkpoint(self):
        logging.info('------>Load pre-trained model from file: %s.', self.pretrained_model_param_path)
        # load model params
        save_dict = self.mxnet_module.nd.load(self.pretrained_model_param_path)

        arg_names = self.net_symbol.list_arguments()
        # get the arg shapes
        arg_name_arrays = {}
        aux_name_arrays = {}

        for k, v in save_dict.items():
            tp, name = k.split(':', 1)
            if name not in arg_names:
                continue
            if tp == 'arg':
                arg_name_arrays.update({name: v.as_in_context(self.mxnet_module.cpu())})
            if tp == 'aux':
                aux_name_arrays.update({name: v.as_in_context(self.mxnet_module.cpu())})
        self.module.init_params(self.net_initializer, arg_name_arrays, aux_name_arrays,
                                allow_missing=True,
                                force_init=True)
