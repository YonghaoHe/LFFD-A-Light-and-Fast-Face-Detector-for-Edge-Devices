# LFFD caffemodel
convert lffd-mxnet-model to lffd-caffe-model

## Introduction
[LFFD: A Light and Fast Face Detector for Edge Devices](https://arxiv.org/abs/1904.10633)



This caffemodel repo is organized as follows:

```
caffemodel
|   |-- README.md
|   |-- configuration_10_320_20L_5scales_v2
|   |   |-- symbol_10_320_20L_5scales_v2_deploy.prototxt
|   |   `-- train_10_320_20L_5scales_v2_iter_1000000.caffemodel
|   |-- configuration_10_560_25L_8scales_v1
|   |   |-- symbol_10_560_25L_8scales_v1_deploy.prototxt
|   |   `-- train_10_560_25L_8scales_v1_iter_1400000.caffemodel
|   |-- predict_caffemodel.py
|   `-- predict_caffemodel_v2.py
```

## Requirements
1. caffe==1.0.0 (I only test on caffe 1.0.0)
2. python==3.6.8
3. Python packages might missing. pls fix it according to the error message.

## mxnet2caffe results
The output feature maps is exactly same between the two models.

## Tips
The different input size should be modified in prototxt.
