"""
pochitrain: A tiny but clever CNN pipeline for images — as friendly as Pochi!

シンプルで親しみやすいCNNパイプラインフレームワーク

Example:
    >>> from pochitrain import Trainer, Config
    >>> config = Config.from_file('configs/resnet/resnet18_cifar10.py')
    >>> trainer = Trainer(config)
    >>> trainer.train()
"""

from .core.config import Config
from .core.trainer import Trainer
from .core.evaluator import Evaluator
from .core.registry import MODELS, DATASETS, TRANSFORMS

__version__ = '0.1.0'
__author__ = 'Pochi Team'
__email__ = 'pochi@example.com'

__all__ = [
    'Config', 'Trainer', 'Evaluator',
    'MODELS', 'DATASETS', 'TRANSFORMS'
]
