## ChasingTrainFramework_GeneralSingleClassDetection
ChasingTrainFramework_GeneralSingleClassDetection is a simple 
wrapper based on MXNet Module API for general one class detection.
`Chasing` is just a project code.

### Framework Introduction
* **data_iterator_base** provide some utils for batch iterator. The design of a data 
iterator relies on the specific task. So we do not provide a default iterator here.

* **data_provider_base** reformat, pack raw data. In most cases, we can load all data into
the memory for fast access.

* **image_augmentation** provide some often used augmentations.

* **inference_speed_eval** provide two ways for inference speed evaluation -- MXNet with CUDNN and TensorRT with CUDNN.

* **loss_layer_farm** provide customized loss type like hard negative mining, focal loss.

* **logging_GOCD** is a logging wrapper.

* **solver_GOCD** execute training process.

* **train_GOCD** is the entrance of the framework.