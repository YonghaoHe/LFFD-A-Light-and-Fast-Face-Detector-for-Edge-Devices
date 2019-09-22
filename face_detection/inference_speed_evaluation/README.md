## Inference Speed Evaluation

### Update History
* `2019.8.1` inference python code for MXNet-cudnn and TensorRT-cudnn is online.

### Additional Prerequirements
* [onnx](https://onnx.ai/) (pip3 install onnx==1.3.0)
* [pycuda](https://developer.nvidia.com/pycuda) (pip3 install pycuda==2019.1.1 or [install guide](https://pypi.org/project/pycuda/))
* [tensorrt](https://developer.nvidia.com/tensorrt) =5.x (use pip3 to install the corresponding .whl file in python folder)

> CAUTION:
>
> Carefully check the version compatible between CUDA, CUDNN, pycuda, TensorRT and onnx.


### Getting Started
1. (optional) temporally add mxnet python path to env if mxnet is not globally set
2. set `eval_with_mxnet_flag` to True to evaluate with mxnet with cudnn, or with tensorrt with cudnn (cannot run both at the same time due to some conflicts)
3. set `symbol_file_path`, `input_shape` and `real_run_loops`
4. run the script