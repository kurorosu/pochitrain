"""
pochitrain: A tiny but clever CNN pipeline for images — as friendly as Pochi!

シンプルで親しみやすいCNNパイプラインフレームワーク

Example:
    >>> from pochitrain import SimpleTrainer, create_data_loaders
    >>> train_loader, val_loader, classes = create_data_loaders('data/train', 'data/val')
    >>> trainer = SimpleTrainer('resnet18', len(classes))
    >>> trainer.setup_training()
    >>> trainer.train(train_loader, val_loader, epochs=50)
"""

# シンプルなインターフェース
from .simple_trainer import SimpleTrainer
from .simple_dataset import (
    SimpleImageDataset,
    create_data_loaders,
    get_basic_transforms,
    print_dataset_info
)
from .models.simple_models import TorchvisionModel, create_model

__version__ = '0.1.0'
__author__ = 'Pochi Team'
__email__ = 'pochi@example.com'

__all__ = [
    # シンプルなインターフェース
    'SimpleTrainer', 'SimpleImageDataset', 'TorchvisionModel',
    'create_data_loaders', 'get_basic_transforms', 'create_model',
    'print_dataset_info'
]
