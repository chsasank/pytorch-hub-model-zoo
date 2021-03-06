# PyTorch Hub Model Zoo

[Pytorch Hub](https://pytorch.org/docs/stable/hub.html) is a pre-trained model repository designed to facilitate research reproducibility. 

It allows you to easily load (pretrained) models from various repositories. For example:

```python-repl
>>> import torch
>>> torch.hub.list('pytorch/vision')
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /Users/Sasank/.cache/torch/hub/master.zip
['alexnet', 'deeplabv3_resnet101', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'fcn_resnet101', 'googlenet', 'inception_v3', 'mobilenet_v2', 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 'resnext101_32x8d', 'resnext50_32x4d', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'wide_resnet101_2', 'wide_resnet50_2']
>>> resnet18 = torch.hub.load('pytorch/vision', 'resnet18')
Using cache found in /Users/Sasank/.cache/torch/hub/pytorch_vision_master
>>> resnet50_pretrained = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
Using cache found in /Users/Sasank/.cache/torch/hub/pytorch_vision_master
Downloading: "https://download.pytorch.org/models/resnet50-19c8e357.pth" to /Users/Sasank/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth
100%|███████████████████████| 97.8M/97.8M [00:16<00:00, 6.28MB/s]

```


Here are the list of models available using torch hub

# Torchvison

* Repository Name: `pytorch/vision` [![Stars](https://img.shields.io/github/stars/pytorch/vision?style=social)](https://github.com/pytorch/vision)
* extra arguments:
    - pretrained (bool): If True, returns a model pre-trained on ImageNet
    - progress (bool): If True, displays a progress bar of the download to stderr
    - Other named arguments that can be passed to original model class.
* Documentation: https://pytorch.org/docs/master/torchvision/models.html


## Image classification models

| Model Name | Description | References |
|---|---|---|
| `alexnet` | AlexNet model architecture from the “One weird trick…” paper. | https://arxiv.org/abs/1404.5997 |
| `vgg11` | VGG 11-layer model (configuration “A”) | https://arxiv.org/abs/1409.1556 |
| `vgg11_bn` | VGG 11-layer model (configuration “A”) with batch normalization | https://arxiv.org/abs/1409.1556 |
| `vgg13` | VGG 11-layer model (configuration “B”) | https://arxiv.org/abs/1409.1556 |
| `vgg13_bn` | VGG 11-layer model (configuration “B”) with batch normalization | https://arxiv.org/abs/1409.1556 |
| `vgg16` | VGG 11-layer model (configuration “D”) | https://arxiv.org/abs/1409.1556 |
| `vgg16_bn` | VGG 11-layer model (configuration “D”) with batch normalization | https://arxiv.org/abs/1409.1556 |
| `vgg19` | VGG 11-layer model (configuration “E”) | https://arxiv.org/abs/1409.1556 |
| `vgg19_bn` | VGG 11-layer model (configuration “E”) with batch normalization | https://arxiv.org/abs/1409.1556 |
| `resnet18` | ResNet-18 mode | https://arxiv.org/abs/1512.03385 |
| `resnet34` | ResNet-34 model | https://arxiv.org/abs/1512.03385 |
| `resnet50` | ResNet-50 model | https://arxiv.org/abs/1512.03385 |
| `resnet101` | ResNet-101 model | https://arxiv.org/abs/1512.03385 |
| `resnet152` | ResNet-152 model | https://arxiv.org/abs/1512.03385 |
| `squeezenet1_0` | SqueezeNet model architecture from the paper | https://arxiv.org/abs/1602.07360 |
| `squeezenet1_1` | SqueezeNet 1.1 model from the official SqueezeNet repo. SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters than SqueezeNet 1.0, without sacrificing accuracy. | https://arxiv.org/abs/1602.07360, [Original Repo](https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1) |
| `densenet121` | Densenet-121 model | https://arxiv.org/abs/1608.06993 |
| `densenet161` | Densenet-161 model | https://arxiv.org/abs/1608.06993 |
| `densenet169` | Densenet-169 model | https://arxiv.org/abs/1608.06993 |
| `densenet201` | Densenet-201 model | https://arxiv.org/abs/1608.06993 |
| `inception_v3` | Inception v3 model architecture | http://arxiv.org/abs/1512.00567 |
| `googlenet` | GoogLeNet (Inception v1) model architecture | http://arxiv.org/abs/1409.4842 |
| `shufflenet_v2_x0_5` | ShuffleNetV2 with 0.5x output channels | https://arxiv.org/abs/1807.11164 |
| `shufflenet_v2_x1_0` | ShuffleNetV2 with 1.0x output channels | https://arxiv.org/abs/1807.11164 |
| `deeplabv3_resnet101` | ShuffleNetV2 with 1.5x output channels | https://arxiv.org/abs/1807.11164 |
| `mobilenet_v2` | MobileNetV2 architecture | https://arxiv.org/abs/1801.04381 |
| `resnext50_32x4d` | ResNeXt-50 32x4d model | https://arxiv.org/abs/1611.05431 |
| `resnext101_32x8d` | ResNeXt-101 32x8d model | https://arxiv.org/abs/1611.05431 |
| `wide_resnet50_2` | Wide ResNet-50-2 model. The model is the same as ResNet except for the bottleneck number of channels which is twice larger in every block. | https://arxiv.org/abs/1605.07146 |
| `wide_resnet101_2` | Wide ResNet-101-2 model. The model is the same as ResNet except for the bottleneck number of channels which is twice larger in every block. | https://arxiv.org/abs/1605.07146 |
| `mnasnet0_5` | MNASNet with depth multiplier of 0.5 | https://arxiv.org/abs/1807.11626 |
| `mnasnet0_75` | MNASNet with depth multiplier of 0.75 | https://arxiv.org/abs/1807.11626 |
| `mnasnet1_0` | MNASNet with depth multiplier of 1.0 | https://arxiv.org/abs/1807.11626 |
| `mnasnet1_3` | MNASNet with depth multiplier of 1.3 | https://arxiv.org/abs/1807.11626 |


## Image Segmentation Models

| Model Name | Reference | References |
|---|---|---|
| `fcn_resnet50` | Fully-Convolutional Network model with a ResNet-50 backbone | https://arxiv.org/abs/1411.4038 |
| `fcn_resnet101` | Fully-Convolutional Network model with a ResNet-101 backbone | https://arxiv.org/abs/1411.4038 |
| `deeplabv3_resnet50` | DeepLabV3 model with a ResNet-50 backbone | https://arxiv.org/abs/1706.05587 |
| `deeplabv3_resnet101` | DeepLabV3 model with a ResNet-101 backbone | https://arxiv.org/abs/1706.05587 |


## Object Detection Models

| Model Name | Reference | References |
|---|---|---|
| `fasterrcnn_resnet50_fpn` | Faster R-CNN model with a ResNet-50-FPN backbone | https://arxiv.org/abs/1612.03144 |
| `maskrcnn_resnet50_fpn` | Mask R-CNN model with a ResNet-50-FPN backbone | https://arxiv.org/abs/1703.06870 |
| `keypointrcnn_resnet50_fpn` | Keypoint R-CNN model with a ResNet-50-FPN backbone |  |


## Example Usage

```python
import torch
mnasnet = torch.hub.load('pytorch/vision', 'mnasnet0_5', pretrained=True)
```

# pretrained-models.pytorch

* Repository Name: `Cadene/pretrained-models.pytorch` [![Stars](https://img.shields.io/github/stars/Cadene/pretrained-models.pytorch?style=social)](https://github.com/Cadene/pretrained-models.pytorch)
* extra arguments:
    - num_classes (int): Number of classes
    - pretrained (str): If None, pretrained model is not loaded. Else have to specify type of pretraining e.g. 'imagenet'.
* Documentation: https://github.com/chsasank/pretrained-models.pytorch/blob/master/README.md


## Image classification models

Models already available in torchvision are excluded in this list.

| Model Name | Description | References |
|---|---|---|
| `nasnetalarge` | NASNet A Large Model | https://arxiv.org/abs/1707.07012, [TensorFlow Slim repo](https://github.com/tensorflow/models/tree/master/research/slim) |
| `nasnetamobile` | NASNet A Mobile Model | https://arxiv.org/abs/1707.07012, [TensorFlow Slim repo](https://github.com/tensorflow/models/tree/master/research/slim) |
| `fbresnet152` | There are a bit different from the ResNet* of torchvision. ResNet152 is currently the only one available. | [Torch7 repo of FaceBook](https://github.com/facebook/fb.resnet.torch) |
| `cafferesnet101` | Resnet101 ported from caffe repo | [Caffe repo of KaimingHe](https://github.com/KaimingHe/deep-residual-networks) |
| `inceptionresnetv2` | Inception v2 Model with Residual connections | https://arxiv.org/abs/1602.07261, [TensorFlow Slim repo](https://github.com/tensorflow/models/tree/master/research/slim) |
| `inceptionv4` | Inception v4 Model | https://arxiv.org/abs/1602.07261, [TensorFlow Slim repo](https://github.com/tensorflow/models/tree/master/research/slim) |
| `bninception` | Inception model from batch norm paper | https://arxiv.org/abs/1502.03167, [Trained with caffe](https://github.com/Cadene/tensorflow-model-zoo.torch/pull/2) |
| `resnext101_32x4d` | ResNeXT 101 32x4d model | https://arxiv.org/abs/1611.05431, [ResNeXt repo of FaceBook](https://github.com/facebookresearch/ResNeXt) |
| `resnext101_62x4d` | ResNeXT 101 62x4d model | https://arxiv.org/abs/1611.05431, [ResNeXt repo of FaceBook](https://github.com/facebookresearch/ResNeXt) |
| `dpn68` | DualPathNetwork 68 | https://arxiv.org/abs/1707.01629, [MXNET repo of Chen Yunpeng](https://github.com/cypw/DPNs) |
| `dpn98` | DualPathNetwork 98 | https://arxiv.org/abs/1707.01629, [MXNET repo of Chen Yunpeng](https://github.com/cypw/DPNs) |
| `dpn131` | DualPathNetwork 131 | https://arxiv.org/abs/1707.01629, [MXNET repo of Chen Yunpeng](https://github.com/cypw/DPNs) |
| `dpn68b` | DualPathNetwork 68b | https://arxiv.org/abs/1707.01629, [MXNET repo of Chen Yunpeng](https://github.com/cypw/DPNs) |
| `dpn92` | DualPathNetwork 92 | https://arxiv.org/abs/1707.01629, [MXNET repo of Chen Yunpeng](https://github.com/cypw/DPNs) |
| `dpn107` | DualPathNetwork 107 | https://arxiv.org/abs/1707.01629, [MXNET repo of Chen Yunpeng](https://github.com/cypw/DPNs) |
| `Xception` | Xception architecture | https://arxiv.org/abs/1610.02357, [Keras repo](https://github.com/keras-team/keras/blob/master/keras/applications/xception.py) |
| `senet154` | SENet 154 | https://arxiv.org/abs/1709.01507, [Caffe repo of Jie Hu](https://github.com/hujie-frank/SENet) |
| `se_resnet50` | Squeeze Excitation (SE) version of resnet50 | https://arxiv.org/abs/1709.01507, [Caffe repo of Jie Hu](https://github.com/hujie-frank/SENet) |
| `se_resnet101` | Squeeze Excitation (SE) version of resnet101 | https://arxiv.org/abs/1709.01507, [Caffe repo of Jie Hu](https://github.com/hujie-frank/SENet) |
| `se_resnet152` | Squeeze Excitation (SE) version of resnet152 | https://arxiv.org/abs/1709.01507, [Caffe repo of Jie Hu](https://github.com/hujie-frank/SENet) |
| `se_resnext50_32x4d` | Squeeze Excitation (SE) version of ResNext50 | https://arxiv.org/abs/1709.01507, [Caffe repo of Jie Hu](https://github.com/hujie-frank/SENet) |
| `se_resnext101_32x4d` | Squeeze Excitation (SE) version of resnet50 | https://arxiv.org/abs/1709.01507, [Caffe repo of Jie Hu](https://github.com/hujie-frank/SENet) |
| `pnasnet5large` | PNASNET 5 Large | https://arxiv.org/abs/1712.00559, [TensorFlow Slim repo](https://github.com/tensorflow/models/tree/master/research/slim) |
| `polynet` | PolyNet | https://arxiv.org/abs/1611.05725, [Caffe repo of the CUHK Multimedia Lab](https://github.com/CUHK-MMLAB/polynet) |


# EfficientNet-PyTorch

* Repository Name: `lukemelas/EfficientNet-PyTorch` [![Stars](https://img.shields.io/github/stars/lukemelas/EfficientNet-PyTorch?style=social)](https://github.com/lukemelas/EfficientNet-PyTorch)
* extra arguments:
    - num_classes (int, optional): Number of classes, default is 1000.
    - in_channels (int, optional): Number of input channels, default is 3.
    - pretrained (str, optional): One of [None, 'imagenet', 'advprop']. If None, no pretrained model is loaded. If 'imagenet', models trained on imagenet dataset are loaded. If 'advprop', models trained using adversarial training called advprop are loaded. It is important to note that the preprocessing required for the advprop pretrained models is slightly different from normal ImageNet preprocessing
* Documentation: https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/README.md

## Image classification models

| Model Name      | Description     | References                       |
|-----------------|-----------------|----------------------------------|
| `efficientnet_b0` | EfficientNet B0 | https://arxiv.org/abs/1905.11946 |
| `efficientnet_b1` | EfficientNet B1 | https://arxiv.org/abs/1905.11946 |
| `efficientnet_b2` | EfficientNet B2 | https://arxiv.org/abs/1905.11946 |
| `efficientnet_b3` | EfficientNet B3 | https://arxiv.org/abs/1905.11946 |
| `efficientnet_b4` | EfficientNet B4 | https://arxiv.org/abs/1905.11946 |
| `efficientnet_b5` | EfficientNet B5 | https://arxiv.org/abs/1905.11946 |
| `efficientnet_b6` | EfficientNet B6 | https://arxiv.org/abs/1905.11946 |
| `efficientnet_b7` | EfficientNet B7 | https://arxiv.org/abs/1905.11946 |
| `efficientnet_b8` | EfficientNet B8 | https://arxiv.org/abs/1905.11946 |


# segmentation_models.pytorch

* Repository Name: `qubvel/segmentation_models.pytorch` [![Stars](https://img.shields.io/github/stars/qubvel/segmentation_models.pytorch?style=social)](https://github.com/qubvel/segmentation_models.pytorch)
* extra arguments:
    - encoder_name (str): name of classification model (without last dense layers) used as feature extractor to build segmentation model.
    - encoder_weights (str): one of `None` (random initialization), `imagenet` (pre-training on ImageNet).
    - See `torch.hub.help` for all the other arguments
* Documentation: https://github.com/qubvel/segmentation_models.pytorch/blob/master/README.md

## Image Segmentation Models

| Model Name | Description | References |
|---|---|---|
| `UNet` | UNet | https://arxiv.org/abs/1505.04597 |
| `Linknet` | LinkNet | https://arxiv.org/abs/1707.03718 |
| `FPN` | FPN (Feature Pyramid Networks) | https://arxiv.org/abs/1612.03144, [presentation](http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf) |
| `PSPNet` | Pyramid Scene Parsing Network (PSPNet) | https://arxiv.org/abs/1612.01105 |
| `PAN` | Pyramid Attention Network (PAN) | https://arxiv.org/abs/1805.10180 |
| `DeepLabV3` | DeepLab v3 | https://arxiv.org/abs/1706.05587 |
| `DeepLabV3Plus` | DeepLab v3+ | https://arxiv.org/abs/1802.02611 |
