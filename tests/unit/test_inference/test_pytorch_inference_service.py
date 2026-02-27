"""PyTorchInferenceService のテスト."""

import logging
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import torch
from torchvision.transforms import Compose, Resize, ToTensor

from pochitrain.config import PochiConfig
from pochitrain.inference.services.pytorch_inference_service import (
    PyTorchInferenceService,
)
from pochitrain.inference.types.orchestration_types import (
    InferenceRuntimeOptions,
)


def _build_logger() -> logging.Logger:
    """テスト用ロガーを返す."""
    logger = logging.getLogger("test_pytorch_inference_service")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.NullHandler())
    return logger


def _build_config(**overrides: Any) -> PochiConfig:
    """テスト用の最小 PochiConfig を返す."""
    defaults: Dict[str, Any] = {
        "model_name": "resnet18",
        "num_classes": 2,
        "device": "cpu",
        "epochs": 1,
        "batch_size": 4,
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "train_data_root": "data/train",
        "val_data_root": "data/val",
        "num_workers": 0,
        "train_transform": Compose([Resize(224), ToTensor()]),
        "val_transform": Compose([Resize(224), ToTensor()]),
        "enable_layer_wise_lr": False,
    }
    defaults.update(overrides)
    return PochiConfig.from_dict(defaults)


# Why:
# 共通ロジックは test_base_inference_service.py で検証済みのため、
# ここでは PyTorch 固有差分のみを検証する.


class TestPyTorchSpecificBehavior:
    """PyTorch 固有差分のテスト."""

    @patch(
        "pochitrain.inference.services.pytorch_inference_service.create_dataset_and_params"
    )
    def test_create_dataloader_delegates_pipeline_strategy(
        self, mock_create_dataset: MagicMock
    ) -> None:
        """データセット戦略の戻り値を保持して DataLoader を構築する."""
        mock_dataset = MagicMock()
        mock_create_dataset.return_value = (
            mock_dataset,
            "fast",
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        )

        service = PyTorchInferenceService(_build_logger())
        config = _build_config()
        data_path = Path("data/val")
        runtime_options = InferenceRuntimeOptions(
            pipeline="gpu",
            batch_size=8,
            num_workers=2,
            pin_memory=False,
            use_gpu=False,
            use_gpu_pipeline=True,
        )
        val_transform = config.val_transform

        loader, dataset, pipeline, norm_mean, norm_std = service.create_dataloader(
            {},
            data_path,
            val_transform,
            "gpu",
            runtime_options,
        )

        mock_create_dataset.assert_called_once_with(
            "gpu",
            data_path,
            val_transform,
        )
        assert dataset is mock_dataset
        assert loader.batch_size == 8
        assert loader.num_workers == 2
        assert loader.pin_memory is False
        assert pipeline == "fast"
        assert norm_mean == [0.485, 0.456, 0.406]
        assert norm_std == [0.229, 0.224, 0.225]

    def test_detect_input_size_uses_transform_then_dataset_fallback(self) -> None:
        """Resize が無い場合はデータセットサンプルへフォールバックする."""
        service = PyTorchInferenceService(_build_logger())

        config = _build_config(val_transform=Compose([Resize(224), ToTensor()]))
        dataset_with_resize = MagicMock()

        config_without_resize = _build_config(val_transform=Compose([ToTensor()]))
        dataset_from_sample = MagicMock()
        dataset_from_sample.__len__ = MagicMock(return_value=1)
        sample_tensor = torch.zeros(3, 100, 100)
        dataset_from_sample.__getitem__ = MagicMock(return_value=(sample_tensor, 0))

        assert service.detect_input_size(config, dataset_with_resize) == (3, 224, 224)
        assert service.detect_input_size(
            config_without_resize, dataset_from_sample
        ) == (3, 100, 100)
