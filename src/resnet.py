#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from typing import Any, List, Type, Union
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
import torch

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

class ResNetHIML(ResNet):
    """ Taken from Biovil
    Wrapper class of the original torchvision ResNet model.
    The forward function is updated to return the penultimate layer
    activations, which are required to obtain image patch embeddings.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.block_ind=3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)

        x1 = self.layer1(x0)
        if self.block_ind == 0:
            return x1

        x2 = self.layer2(x1)
        if self.block_ind == 1:
            return x2

        x3 = self.layer3(x2)
        if self.block_ind == 2:
            return x3

        x4 = self.layer4(x3)
        return x4


def _resnet(arch: str, block: Type[Union[BasicBlock, Bottleneck]], layers: List[int],
            pretrained: bool, progress: bool, **kwargs: Any) -> ResNetHIML:
    """Instantiate a custom :class:`ResNet` model.
    Adapted from :mod:`torchvision.models.resnet`.
    """
    model = ResNetHIML(block=block, layers=layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetHIML:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    :param pretrained: If ``True``, returns a model pre-trained on ImageNet.
    :param progress: If ``True``, displays a progress bar of the download to ``stderr``.
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)

def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetHIML:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    :param pretrained: If ``True``, returns a model pre-trained on ImageNet
    :param progress: If ``True``, displays a progress bar of the download to ``stderr``.
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)
