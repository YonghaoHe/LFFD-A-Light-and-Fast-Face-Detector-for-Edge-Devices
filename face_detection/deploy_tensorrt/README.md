## Deployment with TensorRT
We provide code for deployment with [TensorRT python API](https://developer.nvidia.com/tensorrt).
In general, once you use NVIDIA GPU in your applications, 
TensorRT is the best choice for deployment, rather than training frameworks like TensorFlow, PyTorch, MXNet, Caffe...

### Prerequirements
Refer to [inference_speed_evaluation](../inference_speed_evaluation) for details.

### Getting Started
1. usr `to_onnx.py` to generate onnx model file
2. run `predict_tensorrt.py` to do inference based on the generated model file
3. after you fully understand the code, you may reform and merge it to your own project.

> In most practical cases, C++ is the primary choice for efficient running.
So you can rewrite the code according to the python code structure.
In the future, we will provide C++ version.

### NVIDIA Jetson NANO&TX2 Deployment Instructions
TBD