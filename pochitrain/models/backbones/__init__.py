"""
pochitrain.models.backbones: バックボーンモデル

CNNバックボーンモデルの定義
"""

from .base import BaseBackbone
from .resnet import ResNet
from .vgg import VGG

__all__ = [
    'BaseBackbone', 'ResNet', 'VGG'
]
