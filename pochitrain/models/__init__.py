"""
pochitrain.models: シンプルなモデル定義モジュール

torchvisionモデルを使用したシンプルなモデル定義
"""

from .simple_models import TorchvisionModel, create_model

__all__ = [
    'TorchvisionModel', 'create_model'
]
