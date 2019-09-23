# coding: utf-8
import sys

sys.path.append('/home/heyonghao/libs/incubator-mxnet/python')
import mxnet

eval_with_mxnet_flag = False
symbol_file_path = '/home/heyonghao/projects/ChasingHeadDetection/symbol_farm/symbol_10_160_17L_4scales_v1_deploy.json'
input_shape = (1,3,2160,3840)  # (1,3,240,320) (1,3,480,640) (1,3,720,1280) (1,3,1080,1920) (1,3,2160,3840)
real_run_loops = 200

if eval_with_mxnet_flag:
    from ChasingTrainFramework_GeneralOneClassDetection.inference_speed_eval.inference_speed_eval_with_mxnet_cudnn import InferenceSpeedEval as InferenceSpeedEvalMXNet

    inferenceSpeedEvalMXNet = InferenceSpeedEvalMXNet(symbol_file_path=symbol_file_path, mxnet_module=mxnet, input_shape=input_shape, device_type='gpu', gpu_index=0)
    inferenceSpeedEvalMXNet.run_speed_eval(warm_run_loops=10, real_run_loops=real_run_loops)

else:
    from ChasingTrainFramework_GeneralOneClassDetection.inference_speed_eval.inference_speed_eval_with_tensorrt_cudnn import InferenceSpeedEval as InferenceSpeedEvalTRT

    inferenceSpeedEvalTRT = InferenceSpeedEvalTRT(symbol_file_path=symbol_file_path, mxnet_module=mxnet, input_shape=input_shape)
    inferenceSpeedEvalTRT.run_speed_eval(warm_run_loops=10, real_run_loops=real_run_loops)
