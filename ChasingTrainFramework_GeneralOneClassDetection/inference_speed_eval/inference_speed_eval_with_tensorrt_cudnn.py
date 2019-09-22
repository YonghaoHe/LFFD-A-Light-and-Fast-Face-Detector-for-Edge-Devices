# coding: utf-8
import sys
import os
import time
import logging
import numpy

import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

logging.getLogger().setLevel(logging.INFO)


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class InferenceSpeedEval(object):
    def __init__(self, symbol_file_path, mxnet_module, input_shape, data_mode='fp32'):

        if not os.path.exists(symbol_file_path):
            logging.error('symbol file does not exist!')
            sys.exit(1)

        if len(input_shape) != 4:
            logging.error('input shape should have 4 elements in the order of NCHW.')
            sys.exit(1)

        symbol_net = mxnet_module.symbol.load(symbol_file_path)
        # create module
        module = mxnet_module.module.Module(symbol=symbol_net,
                                            data_names=['data'],
                                            label_names=None,
                                            context=mxnet_module.cpu())
        module.bind(data_shapes=[('data', input_shape)], for_training=False, grad_req='write')
        module.init_params(initializer=mxnet_module.initializer.Xavier(), allow_missing=True)
        arg_params, aux_params = module.get_params()
        net_params = dict()
        net_params.update(arg_params)
        net_params.update(aux_params)
        self.onnx_temp_file = 'temp.onnx'
        logging.info('Convert mxnet symbol to onnx...')
        mxnet_module.contrib.onnx.export_model(symbol_net, net_params, [input_shape], numpy.float32, self.onnx_temp_file, verbose=False)

        # build engine
        trt_logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(trt_logger)
        builder.max_batch_size = input_shape[0]
        builder.average_find_iterations = 2
        builder.max_workspace_size = 2 << 30

        if data_mode == 'fp32':
            pass
        elif data_mode == 'fp16':
            if not builder.platform_has_fast_fp16:
                logging.error('fp16 is not supported by this platform!')
                sys.exit(1)
            builder.fp16_mode = True
        elif data_mode == 'int8':
            logging.error('Currently, not implemented yet.')
            sys.exit(1)
            if not builder.platform_has_fast_int8:
                logging.error('int8 is not supported by this platform!')
                sys.exit(1)
            builder.int8_mode = True
        else:
            logging.error('Unknown data_mode: %s' % data_mode)
            logging.error('Available choices: \'fp32\'(default), \'fp16\', \'int8\'')
            sys.exit(1)

        network = builder.create_network()
        parser = trt.OnnxParser(network, trt_logger)
        logging.info('Parsing onnx for trt network...')
        with open(self.onnx_temp_file, 'rb') as onnx_fin:
            parser.parse(onnx_fin.read())

        num_parser_errors = parser.num_errors
        if num_parser_errors != 0:
            logging.error('Errors occur while parsing the onnx file!')
            for i in range(num_parser_errors):
                logging.error('Error %d: %s' % (i, parser.get_error(i).desc()))
            sys.exit(1)

        logging.info('Start to build trt engine...(this step may cost much time)')
        time_start = time.time()
        self.engine = builder.build_cuda_engine(network)
        time_end = time.time()
        logging.info('Engine building time: %.02f s' % (time_end - time_start))

        for binding in self.engine:
            if self.engine.binding_is_input(binding):
                logging.info('Input name: %s, shape: %s' % (binding, str(self.engine.get_binding_shape(binding))))

        self.executor = self.engine.create_execution_context()
        self.max_batch_size = builder.max_batch_size

    def __del__(self):
        if os.path.exists(self.onnx_temp_file):
            os.remove(self.onnx_temp_file)

    def run_speed_eval(self, warm_run_loops=10, real_run_loops=100):

        def allocate_buffers(engine):
            inputs = []
            outputs = []
            bindings = []
            for binding in engine:
                size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
                dtype = trt.nptype(engine.get_binding_dtype(binding))
                # Allocate host and device buffers
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                # Append the device buffer to device bindings.
                bindings.append(int(device_mem))
                # Append to the appropriate list.
                if engine.binding_is_input(binding):
                    inputs.append(HostDeviceMem(host_mem, device_mem))
                else:
                    outputs.append(HostDeviceMem(host_mem, device_mem))
            return inputs, outputs, bindings

        inputs, outputs, bindings = allocate_buffers(self.engine)
        # warm run
        for i in range(warm_run_loops):
            [cuda.memcpy_htod(inp.device, inp.host) for inp in inputs]
            self.executor.execute(batch_size=self.max_batch_size, bindings=bindings)
            [cuda.memcpy_dtoh(out.host, out.device) for out in outputs]

        # real run
        logging.info('Start real run loop.')
        sum_time_data_copy = 0.
        sum_time_inference_only = 0.
        for i in range(real_run_loops):
            time_start = time.time()
            [cuda.memcpy_htod(inp.device, inp.host) for inp in inputs]
            sum_time_data_copy += time.time() - time_start

            time_start = time.time()
            self.executor.execute(batch_size=self.max_batch_size, bindings=bindings)
            sum_time_inference_only += time.time() - time_start

            time_start = time.time()
            [cuda.memcpy_dtoh(out.host, out.device) for out in outputs]
            sum_time_data_copy += time.time() - time_start

        logging.info('Total time (data transfer & inference) elapsed: %.02f ms. [%.02f ms] for each image (%.02f PFS)'
                     % ((sum_time_data_copy + sum_time_inference_only) * 1000,
                        (sum_time_data_copy + sum_time_inference_only) * 1000 / real_run_loops / self.max_batch_size,
                        real_run_loops * self.max_batch_size / (sum_time_data_copy + sum_time_inference_only)))


if __name__ == '__main__':
    sys.path.append('/home/heyonghao/libs/incubator-mxnet/python')
    import mxnet

    symbol_file_path = '/home/heyonghao/projects/tocreate_LFFD_ICCV2019_FaceDetector/symbol_farm/symbol_10_560_25L_8scales_s5_v2_deploy.json'
    input_shape = (1, 3, 720, 1280)  # (1, 3, 240, 320) (1, 3, 480, 640) (1, 3, 720, 1280) (1, 3, 1080, 1920) (1, 3, 2160, 3840) (1, 3, 4320, 7680)

    speedEval = InferenceSpeedEval(symbol_file_path=symbol_file_path, mxnet_module=mxnet, input_shape=input_shape, data_mode='fp32')
    speedEval.run_speed_eval(warm_run_loops=10, real_run_loops=500)
