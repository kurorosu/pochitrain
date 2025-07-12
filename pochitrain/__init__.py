"""
pochitrain: A tiny but clever CNN pipeline for images — as friendly as Pochi!

シンプルで親しみやすいCNNパイプラインフレームワーク

Example:
    >>> from pochitrain import PochiTrainer, create_data_loaders
    >>> train_loader, val_loader, classes = create_data_loaders('data/train', 'data/val')
    >>> trainer = PochiTrainer('resnet18', len(classes))
    >>> trainer.setup_training()
    >>> trainer.train(train_loader, val_loader, epochs=50)
"""

# Pochiインターフェース
from .pochi_trainer import PochiTrainer
from .pochi_dataset import (
    PochiImageDataset,
    create_data_loaders,
    get_basic_transforms,
    print_dataset_info
)
from .models.pochi_models import PochiModel, create_model

__version__ = '0.1.0'
__author__ = 'Pochi Team'
__email__ = 'pochi@example.com'

__all__ = [
    # Pochiインターフェース
    'PochiTrainer', 'PochiImageDataset', 'PochiModel',
    'create_data_loaders', 'get_basic_transforms', 'create_model',
    'print_dataset_info'
]
