# A Light and Fast Face Detector for Edge Devices
**This repo is updated frequently, keeping up with the latest code is highly recommended.**

## Recent Update
* `2019.07.25` This repos is first online. Face detection code and trained models are released.
* `2019.08.15` This repos is formally released. Any advice and error reports are sincerely welcome.
* `2019.08.22` face_detection: latency evaluation on TX2 is added.
* `2019.08.25` face_detection: RetinaFace-MobileNet-0.25 is added for comparison (both accuracy and latency).
* `2019.09.09` LFFD is ported to NCNN ([link](https://github.com/SyGoing/LFFD-with-ncnn)) and MNN ([link](https://github.com/SyGoing/LFFD-MNN)) by [SyGoing](https://github.com/SyGoing), great thanks to SyGoing.
* `2019.09.10` face_detection: **important bug fix:** vibration offset should be subtracted by shift in data iterator. This bug may result in lower accuracy, inaccurate bbox prediction and bbox vibration in test phase.
We will upgrade v1 and v2 as soon as possible (should have higher accuracy and more stable).
* `2019.09.17` face_detection: model v2 is upgraded! After fixing the bug, we have fine-tuned the old v2 model. The accuracy on 
WIDER FACE is improved significantly! Please try new v2.
* `2019.09.18` pedestrian_detection: preview version of model v1 for Caltech Pedestrian Dataset is released.
* `2019.09.23` head_detection: model v1 for brainwash dataset is released.
* `2019.10.02` license_plate_detection: model v1 for CCPD dataset is released. (**The accuracy is very high and the latency is very short!** Have a try.)
* `2019.10.02` Currently, we have provided some application-oriented detectors. Subsequently, we will put most energy to 
next generation framework for single-class detection. Any feedback is welcome.
* `2019.10.16` face_detection: the preview of PyTorch version is ready ([link](https://github.com/becauseofAI/lffd-pytorch)). Any feedback is welcome.
* `2019.10.16` Tips: data preparation is important, irrational values of (x,y,w,h) may introduce nan in training; we
trained models with convs followed by BNs. But we found that the convergence is not stable, and can not reach a good point.

* `2019.07.25` This repos is first online. Face detection code and trained models are released.
* `2019.08.15` This repos is formally released. Any advice and error reports are sincerely welcome.
* `2019.08.22` face_detection: latency evaluation on TX2 is added.
* `2019.08.25` face_detection: RetinaFace-MobileNet-0.25 is added for comparison (both accuracy and latency).
* `2019.09.09` LFFD is ported to NCNN ([link](https://github.com/SyGoing/LFFD-with-ncnn)) and MNN ([link](https://github.com/SyGoing/LFFD-MNN)) by [SyGoing](https://github.com/SyGoing), great thanks to SyGoing.
* `2019.09.10` face_detection: **important bug fix:** vibration offset should be subtracted by shift in data iterator. This bug may result in lower accuracy, inaccurate bbox prediction and bbox vibration in test phase.
We will upgrade v1 and v2 as soon as possible (should have higher accuracy and more stable).
* `2019.09.17` face_detection: model v2 is upgraded! After fixing the bug, we have fine-tuned the old v2 model. The accuracy on 
WIDER FACE is improved significantly! Please try new v2.
* `2019.09.18` pedestrian_detection: preview version of model v1 for Caltech Pedestrian Dataset is released.
* `2019.09.23` head_detection: model v1 for brainwash dataset is released.
* `2019.10.02` license_plate_detection: model v1 for CCPD dataset is released. (**The accuracy is very high and the latency is very short!** Have a try.)
* `2019.10.02` Currently, we have provided some application-oriented detectors. Subsequently, we will put most energy to 
next generation framework for single-class detection. Any feedback is welcome.
* `2019.10.16` face_detection: the preview of PyTorch version is ready ([link](https://github.com/becauseofAI/lffd-pytorch)). Any feedback is welcome.
* `2019.10.16` Tips: data preparation is important, irrational values of (x,y,w,h) may introduce nan in training; we
trained models with convs followed by BNs. But we found that the convergence is not stable, and can not reach a good point.

## Introduction
This repo releases the source code of paper "[LFFD: A Light and Fast Face Detector for Edge Devices](https://arxiv.org/abs/1904.10633)". Our paper presents a light and fast face detector (**LFFD**) for edge devices.
LFFD considerably balances both accuracy and latency, resulting in small model size, fast inference speed while achieving excellent accuracy.
**Understanding the essence of receptive field makes detection networks interpretable.**
  
In practical, we have deployed it in cloud and edge devices (like NVIDIA Jetson series and ARM-based embedding system). The comprehensive performance
of LFFD is robust enough to support our applications.

In fact, our method is **_a general detection framework that applicable to one class detection_**, such as face detection, pedestrian detection, 
head detection, vehicle detection and so on. In general, an object class, whose average ratio of the longer side and the shorter side is 
less than 5, is appropriate to apply our framework for detection.

Several practical advantages:
1. large scale coverage, and easy to extend to larger scales by adding more layers without much latency gain.
2. detect small objects (as small as 10 pixels) in images with extremely large resolution (8K or even larger) in only one inference.
3. easy backbone with very common operators makes it easy to deploy anywhere.

## Accuracy and Latency
We train LFFD on train set of WIDER FACE benchmark. All methods are evaluated on val/test sets under the SIO schema (please
refer to the paper for details).

* Accuracy on val set of WIDER FACE (The values in () are results from the original papers):

Method|Easy Set|Medium Set|Hard Set
------|--------|----------|--------
DSFD  |0.949(0.966)|0.936(0.957)|0.850(0.904)
PyramidBox|0.937(0.961)|0.927(0.950)|0.867(0.889)
S3FD  |0.923(0.937)|0.907(0.924)|0.822(0.852)
SSH   |0.921(0.931)|0.907(0.921)|0.702(0.845)
FaceBoxes|0.840    |0.766       |0.395
FaceBoxes3.2×|0.798|0.802       |0.715
**LFFD**|0.910     |0.881       |0.780

* Accuracy on test set of WIDER FACE (The values in () are results from the original papers):

Method|Easy Set|Medium Set|Hard Set
------|--------|----------|--------
DSFD  |0.947(0.960)|0.934(0.953)|0.845(0.900)
PyramidBox|0.926(0.956)|0.920(0.946)|0.862(0.887)
S3FD  |0.917(0.928)|0.904(0.913)|0.821(0.840)
SSH   |0.919(0.927)|0.903(0.915)|0.705(0.844)
FaceBoxes|0.839    |0.763       |0.396
FaceBoxes3.2×|0.791|0.794       |0.715
**LFFD**|0.896     |0.865       |0.770

* Accuracy on FDDB:

Method|Disc ROC curves score
------|--------
DFSD|0.984
PyramidBox|0.982
S3FD|0.981
SSH|0.977
FaceBoxes3.2×|0.905
FaceBoxes|0.960
LFFD|0.973

In the paper, three hardware platforms are used for latency evaluation: NVIDIA GTX TITAN Xp, NVIDIA TX2 and 
Rasberry Pi 3 Model B+ (ARM A53). 

We report the latency of inference only (for NVIDIA hardwares, data transfer is included), excluding
pre-processing and post-processing. The batchsize is set to 1 for all evaluations.

* Latency on NVIDIA GTX TITAN Xp (MXNet+CUDA 9.0+CUDNN7.1):

Resolution->|640×480|1280×720|1920×1080|3840×2160
------------|-------|--------|---------|---------
DSFD|78.08ms(12.81 FPS)|187.78ms(5.33 FPS)|392.82ms(2.55 FPS)|1562.50ms(0.64 FPS)
PyramidBox|50.51ms(19.08 FPS)|143.34ms(6.98 FPS)|331.93ms(3.01 FPS)|1344.07ms(0.74 FPS)
S3FD|21.75ms(45.95 FPS)|55.73ms(17.94 FPS)|119.53ms(8.37 FPS)|471.31ms(2.21 FPS)
SSH|22.44ms(44.47 FPS)|55.29ms(18.09 FPS)|118.43ms(8.44 FPS)|463.10ms(2.16 FPS)
FaceBoxes3.2×|6.80ms(147.00 FPS)|12.96ms(77.19 FPS)|25.37ms(39.41 FPS)|111.98ms(8.93 FPS)
**LFFD**|7.60ms(131.40 FPS)|16.37ms(61.07 FPS)|31.27ms(31.98 FPS)|87.79ms(11.39 FPS)

* Latency on NVIDIA TX2 (MXNet+CUDA 9.0+CUDNN7.1) presented in the paper:

Resolution->|160×120|320×240|640×480
------------|-------|--------|---------
FaceBoxes3.2×|11.20ms(89.29 FPS)|19.62ms(50.97 FPS)|72.74ms(13.75 FPS)
**LFFD**|7.30ms(136.99 FPS)|19.64ms(50.92 FPS)|64.70ms(15.46 FPS)

* Latency on Respberry Pi 3 Model B+ (ncnn) presented in the paper:

Resolution->|160×120|320×240|640×480
------------|-------|--------|---------
FaceBoxes3.2×|167.20ms(5.98 FPS)|686.19ms(1.46 FPS)|3232.26ms(0.31 FPS)
**LFFD**|118.45ms(8.44 FPS)|409.19ms(2.44 FPS)|4114.15ms(0.24 FPS)

> On NVIDIA platform, TensorRT is the best choice for inference. So we conduct additional latency evaluations using TensorRT 
(**the latency is dramatically decreased!!!**). 
As for ARM based platform, we plan to use [MNN](https://github.com/alibaba/MNN) and [Tengine](https://github.com/OAID/Tengine) 
for latency evaluation. Details can be found in the sub-project [face_detection](face_detection/README.md).

## Getting Started
We implement the proposed method using MXNet Module API.

#### Prerequirements (global)
* Python>=3.5
* numpy>=1.16 (lower versions should work as well, but not tested)
* MXNet>=1.4.1 ([install guide](http://mxnet.incubator.apache.org/versions/master/install/index.html?platform=Linux&language=Python&processor=GPU))
* cv2=3.x (pip3 install opencv-python==3.4.5.20, other version should work as well, but not tested)

> Tips: 
  * use MXNet with cudnn.
  * build numpy from source with OpenBLAS. This will improve the training efficiency.
  * make sure cv2 links to libjpeg-turbo, not libjpeg. This will improve the jpeg decode efficiency.

#### Sub-directory description
* [face_detection](face_detection) contains the code of training, evaluation and inference for LFFD,
the main content of this repo. The trained models of different versions are provided for off-the-shelf deployment.
* [head_detection](head_detection) contains the trained models for head detection. The models are obtained by the
proposed general one class detection framework.
* [pedestrian_detection](pedestrian_detection) contains the trained models for pedestrian detection. The models are obtained by the
proposed general one class detection framework.
* [vehicle_detection](vehicle_detection) contains the trained models for vehicle detection. The models are obtained by the
proposed general one class detection framework.
* [ChasingTrainFramework_GeneralOneClassDetection](ChasingTrainFramework_GeneralOneClassDetection) is a simple 
wrapper based on MXNet Module API for general one class detection.

#### Installation
1. Download the repo:
```
git clone https://github.com/YonghaoHe/A-Light-and-Fast-Face-Detector-for-Edge-Devices.git
```
2. Refer to the corresponding sub-project for detailed usage.

## Citation
If you benefit from our work in your research and product, please kindly cite the paper
```
@inproceedings{LFFD,
title={LFFD: A Light and Fast Face Detector for Edge Devices},
author={He, Yonghao and Xu, Dezhong and Wu, Lifang and Jian, Meng and Xiang, Shiming and Pan, Chunhong},
booktitle={arXiv:1904.10633},
year={2019}
}
```

## To Do List
- [x] face detection
- [x] pedestrian detection
- [x] head detection
- [ ] vehicle detection
- [x] license plate detection
- [x] [PyTorch version (becauseofAI)](https://github.com/becauseofAI/lffd-pytorch)

## Contact
Yonghao He

E-mails: yonghao.he@ia.ac.cn / yonghao.he@aliyun.com

**If you are interested in this work, any innovative contributions are welcome!!!**

**Internship is open at NLPR, CASIA all the time. Send me your resumes!**
