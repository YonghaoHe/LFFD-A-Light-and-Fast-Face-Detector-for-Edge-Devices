# -*- coding: utf-8 -*-
import logging


def start_train(param_dict,
                mxnet_module,
                context,
                train_dataiter,
                train_metric,
                train_metric_update_frequency,
                num_train_loops,
                val_dataiter,
                val_metric,
                num_val_loops,
                validation_interval,
                optimizer_name,
                optimizer_params,
                net_symbol,
                net_initializer,
                net_data_names,
                net_label_names,
                pretrained_model_param_path,
                display_interval,
                save_prefix,
                model_save_interval,
                start_index
                ):

    logging.info('MXNet Version: %s', str(mxnet_module.__version__))
    logging.info('Training settings:-----------------------------------------------------------------')
    for param_name, param_value in param_dict.items():
        logging.info(param_name + ':' + str(param_value))
    logging.info('-----------------------------------------------------------------------------------')

    # init Solver module-------------------------------------------------------------------------------------
    from .solver_GOCD import Solver

    solver = Solver(
        mxnet_module=mxnet_module,
        trainset_dataiter=train_dataiter,
        net_symbol=net_symbol,
        net_initializer=net_initializer,
        optimizer_name=optimizer_name,
        optimizer_params=optimizer_params,
        data_names=net_data_names,
        label_names=net_label_names,
        context=context,
        num_train_loops=num_train_loops,
        train_metric=train_metric,
        display_interval=display_interval,
        val_evaluation_interval=validation_interval,
        valset_dataiter=val_dataiter,
        val_metric=val_metric,
        num_val_loops=num_val_loops,
        pretrained_model_param_path=pretrained_model_param_path,
        save_prefix=save_prefix,
        start_index=start_index,
        model_save_interval=model_save_interval,
        train_metric_update_frequency=train_metric_update_frequency)
    solver.fit()
