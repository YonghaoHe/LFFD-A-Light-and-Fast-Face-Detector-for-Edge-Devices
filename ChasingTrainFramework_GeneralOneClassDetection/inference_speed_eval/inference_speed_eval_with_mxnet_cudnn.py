# coding: utf-8
import sys
import os
import time
import logging

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '1'
logging.getLogger().setLevel(logging.INFO)


class InferenceSpeedEval(object):
    def __init__(self, symbol_file_path, mxnet_module, input_shape, input_name='data', device_type='gpu', gpu_index=0):
        '''

        :param symbol_file_path: symbol file path
        :param mxnet_module: mxnet module
        :param input_shape: input shape in tuple--(batch_size, num_channel, height, width)
        :param input_name: input name defined in symbol, by default 'data'
        :param device_type: device type: 'gpu', 'cpu'
        :param gpu_index: gpu index
        '''
        self.symbol_file_path = symbol_file_path
        self.mxnet_module = mxnet_module
        self.input_name = input_name
        self.input_shape = input_shape
        self.device_type = device_type
        if self.device_type == 'cpu':  # CAUTION: x86 cpu inference needs MXNet with mkldnn, or inference speed will be very slow
            self.context = self.mxnet_module.cpu()
        elif self.device_type == 'gpu':
            self.context = self.mxnet_module.gpu(gpu_index)
        else:
            logging.error('Unknow device_type: %s .' % self.device_type)
            sys.exit(1)

        # load symbol file
        if not os.path.exists(self.symbol_file_path):
            logging.error('Symbol file: %s does not exist!' % symbol_file_path)
            sys.exit(1)
        self.symbol_net = self.mxnet_module.symbol.load(self.symbol_file_path)

        # create module
        self.module = self.mxnet_module.module.Module(symbol=self.symbol_net,
                                                      data_names=[self.input_name],
                                                      label_names=None,
                                                      context=self.context)
        self.module.bind(data_shapes=[(self.input_name, self.input_shape)], for_training=False, grad_req='write')

        self.module.init_params(initializer=self.mxnet_module.initializer.Xavier(), allow_missing=True)
        self.module.init_optimizer(kvstore=None)

    def run_speed_eval(self, warm_run_loops=10, real_run_loops=100):
        random_input_data = [self.mxnet_module.random.uniform(-1.0, 1.0, shape=self.input_shape, ctx=self.context)]
        temp_batch = self.mxnet_module.io.DataBatch(random_input_data, [])

        # basic info of this eval
        logging.info('Test symbol file: %s' % self.symbol_file_path)
        logging.info('Test device: %s' % self.device_type)
        logging.info('Test input shape: %s' % str(self.input_shape))

        # warn run
        for i in range(warm_run_loops):
            self.module.forward(temp_batch)
            for output in self.module.get_outputs():
                output.asnumpy()

        logging.info('Start real run loops---------------')
        tic = time.time()
        # real run
        for i in range(real_run_loops):
            self.module.forward(temp_batch)
            for output in self.module.get_outputs():
                output.asnumpy()

        toc = time.time()

        print('Finish %d loops in %.02f ms. \n[%.02f ms] for each loop \n[%.02f ms] for each image (namely %.02f FPS)' %
              (real_run_loops,
               (toc - tic) * 1000,
               (toc - tic) * 1000 / real_run_loops,
               (toc - tic) * 1000 / real_run_loops / self.input_shape[0],
               real_run_loops * self.input_shape[0] / (toc - tic)))


if __name__ == '__main__':
    sys.path.append('/home/heyonghao/libs/incubator-mxnet/python')  # set MXNet python path if needed
    import mxnet

    symbol_file_path = '/home/heyonghao/projects/tocreate_LFFD_ICCV2019_FaceDetector/symbol_farm/symbol_10_560_25L_8scales_s5_v2_deploy.json'
    input_shape = (1, 3, 720, 1280)  # (1, 3, 240, 320) (1, 3, 480, 640) (1, 3, 720, 1280) (1, 3, 1080, 1920) (1, 3, 2160, 3840) (1, 3, 4320, 7680)
    device_type = 'gpu'
    gpu_index = 0

    speedEval = InferenceSpeedEval(symbol_file_path=symbol_file_path, mxnet_module=mxnet, input_shape=input_shape, device_type=device_type, gpu_index=gpu_index)
    speedEval.run_speed_eval(warm_run_loops=10, real_run_loops=500)
