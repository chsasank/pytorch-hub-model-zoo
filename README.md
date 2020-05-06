# PyTorch Hub Model Zoo

[Pytorch Hub](https://pytorch.org/docs/stable/hub.html) is a pre-trained model repository designed to facilitate research reproducibility. 

It allows you to easily load (pretrained) models from various repositories. For example:

```python
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

* Repository Name: `pytorch/vision`
* extra arguments:
    - pretrained (bool): If True, returns a model pre-trained on ImageNet
    - progress (bool): If True, displays a progress bar of the download to stderr
    - Other named arguments that can be passed to original model class.
* Documentation: https://pytorch.org/docs/master/torchvision/models.html


## Image classification models

| Model Name | Description | References |
|---|---|---|
| alexnet | AlexNet model architecture from the “One weird trick…” paper. | https://arxiv.org/abs/1404.5997 |
| vgg11 | VGG 11-layer model (configuration “A”) | https://arxiv.org/abs/1409.1556 |
| vgg11_bn | VGG 11-layer model (configuration “A”) with batch normalization | https://arxiv.org/abs/1409.1556 |
| vgg13 | VGG 11-layer model (configuration “B”) | https://arxiv.org/abs/1409.1556 |
| vgg13_bn | VGG 11-layer model (configuration “B”) with batch normalization | https://arxiv.org/abs/1409.1556 |
| vgg16 | VGG 11-layer model (configuration “D”) | https://arxiv.org/abs/1409.1556 |
| vgg16_bn | VGG 11-layer model (configuration “D”) with batch normalization | https://arxiv.org/abs/1409.1556 |
| vgg19 | VGG 11-layer model (configuration “E”) | https://arxiv.org/abs/1409.1556 |
| vgg19_bn | VGG 11-layer model (configuration “E”) with batch normalization | https://arxiv.org/abs/1409.1556 |
| resnet18 | ResNet-18 mode | https://arxiv.org/abs/1512.03385 |
| resnet34 | ResNet-34 model | https://arxiv.org/abs/1512.03385 |
| resnet50 | ResNet-50 model | https://arxiv.org/abs/1512.03385 |
| resnet101 | ResNet-101 model | https://arxiv.org/abs/1512.03385 |
| resnet152 | ResNet-152 model | https://arxiv.org/abs/1512.03385 |
| squeezenet1_0 | SqueezeNet model architecture from the paper | https://arxiv.org/abs/1602.07360 |
| squeezenet1_1 | SqueezeNet 1.1 model from the official SqueezeNet repo. SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters than SqueezeNet 1.0, without sacrificing accuracy. | https://arxiv.org/abs/1602.07360, https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1 |
| densenet121 | Densenet-121 model | https://arxiv.org/abs/1608.06993 |
| densenet161 | Densenet-161 model | https://arxiv.org/abs/1608.06993 |
| densenet169 | Densenet-169 model | https://arxiv.org/abs/1608.06993 |
| densenet201 | Densenet-201 model | https://arxiv.org/abs/1608.06993 |
| inception_v3 | Inception v3 model architecture | http://arxiv.org/abs/1512.00567 |
| googlenet | GoogLeNet (Inception v1) model architecture | http://arxiv.org/abs/1409.4842 |
| shufflenet_v2_x0_5 | ShuffleNetV2 with 0.5x output channels | https://arxiv.org/abs/1807.11164 |
| shufflenet_v2_x1_0 | ShuffleNetV2 with 1.0x output channels | https://arxiv.org/abs/1807.11164 |
| deeplabv3_resnet101 | ShuffleNetV2 with 1.5x output channels | https://arxiv.org/abs/1807.11164 |
| mobilenet_v2 | MobileNetV2 architecture | https://arxiv.org/abs/1801.04381 |
| resnext50_32x4d | ResNeXt-50 32x4d model | https://arxiv.org/abs/1611.05431 |
| resnext101_32x8d | ResNeXt-101 32x8d model | https://arxiv.org/abs/1611.05431 |
| wide_resnet50_2 | Wide ResNet-50-2 model. The model is the same as ResNet except for the bottleneck number of channels which is twice larger in every block. | https://arxiv.org/abs/1605.07146 |
| wide_resnet101_2 | Wide ResNet-101-2 model. The model is the same as ResNet except for the bottleneck number of channels which is twice larger in every block. | https://arxiv.org/abs/1605.07146 |
| mnasnet0_5 | MNASNet with depth multiplier of 0.5 | https://arxiv.org/abs/1807.11626 |
| mnasnet0_75 | MNASNet with depth multiplier of 0.75 | https://arxiv.org/abs/1807.11626 |
| mnasnet1_0 | MNASNet with depth multiplier of 1.0 | https://arxiv.org/abs/1807.11626 |
| mnasnet1_3 | MNASNet with depth multiplier of 1.3 | https://arxiv.org/abs/1807.11626 |