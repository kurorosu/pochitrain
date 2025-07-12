"""
pochitrain.models: Pochiモデル定義モジュール

torchvisionモデルを使用したPochiモデル定義
"""

from .pochi_models import PochiModel, create_model

__all__ = [
    'PochiModel', 'create_model'
]
