"""
pochitrain: 画像分類向けの軽量 CNN 学習パイプライン.

Example:
    >>> from pochitrain import PochiTrainer, create_data_loaders
    >>> train_loader, val_loader, classes = create_data_loaders("data/train", "data/val")  # noqa: E501
    >>> trainer = PochiTrainer("resnet18", len(classes))
    >>> trainer.setup_training()
    >>> trainer.train(train_loader, val_loader, epochs=50)
"""

from .config import PochiConfig

# Logging
from .logging import LoggerManager
from .models.pochi_models import PochiModel, create_model
from .pochi_dataset import (
    PochiImageDataset,
    create_data_loaders,
    get_basic_transforms,
    print_dataset_info,
)
from .pochi_predictor import PochiPredictor

# Pochiインターフェース
from .pochi_trainer import PochiTrainer

# ユーティリティ
from .utils.directory_manager import InferenceWorkspaceManager, PochiWorkspaceManager

__version__ = "1.3.0"
__author__ = "Pochi Team"
__email__ = "pochi@example.com"

__all__ = [
    # Logging
    "LoggerManager",
    # Pochiインターフェース
    "PochiTrainer",
    "PochiPredictor",
    "PochiConfig",
    "PochiImageDataset",
    "PochiModel",
    "create_data_loaders",
    "get_basic_transforms",
    "create_model",
    "print_dataset_info",
    # ユーティリティ
    "PochiWorkspaceManager",
    "InferenceWorkspaceManager",
]
