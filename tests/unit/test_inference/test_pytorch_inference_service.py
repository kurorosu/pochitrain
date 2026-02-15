"""PyTorchInferenceService のテスト."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest
import torch
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor

from pochitrain.config import PochiConfig
from pochitrain.inference.services.pytorch_inference_service import (
    PyTorchInferenceService,
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


class TestCreatePredictor:
    """create_predictor のテスト."""

    @patch("pochitrain.inference.services.pytorch_inference_service.PochiPredictor")
    def test_delegates_to_from_config(self, mock_predictor_cls: MagicMock) -> None:
        """PochiPredictor.from_config に正しく委譲されること."""
        service = PyTorchInferenceService(_build_logger())
        config = _build_config()
        model_path = Path("models/best.pth")

        service.create_predictor(config, model_path)

        mock_predictor_cls.from_config.assert_called_once_with(config, str(model_path))


class TestCreateDataloader:
    """create_dataloader のテスト."""

    @patch("pochitrain.inference.services.pytorch_inference_service.PochiImageDataset")
    def test_returns_loader_and_dataset(self, mock_dataset_cls: MagicMock) -> None:
        """DataLoader とデータセットが返されること."""
        mock_dataset = MagicMock()
        mock_dataset_cls.return_value = mock_dataset

        service = PyTorchInferenceService(_build_logger())
        config = _build_config()
        data_path = Path("data/val")

        loader, dataset = service.create_dataloader(config, data_path)

        mock_dataset_cls.assert_called_once_with(
            str(data_path), transform=config.val_transform
        )
        assert dataset is mock_dataset
        assert loader.batch_size == config.batch_size


class TestDetectInputSize:
    """detect_input_size のテスト."""

    def test_from_resize_transform(self) -> None:
        """Resize Transform からサイズを取得できること."""
        config = _build_config(val_transform=Compose([Resize(224), ToTensor()]))
        dataset = MagicMock()
        service = PyTorchInferenceService(_build_logger())

        result = service.detect_input_size(config, dataset)

        assert result == (3, 224, 224)

    def test_from_center_crop_transform(self) -> None:
        """CenterCrop Transform からサイズを取得できること."""
        config = _build_config(
            val_transform=Compose([CenterCrop((128, 256)), ToTensor()])
        )
        dataset = MagicMock()
        service = PyTorchInferenceService(_build_logger())

        result = service.detect_input_size(config, dataset)

        assert result == (3, 128, 256)

    def test_fallback_to_dataset_sample(self) -> None:
        """Transform にサイズ情報がない場合, データセットからフォールバックすること."""
        config = _build_config(val_transform=Compose([ToTensor()]))
        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=1)
        sample_tensor = torch.zeros(3, 100, 100)
        dataset.__getitem__ = MagicMock(return_value=(sample_tensor, 0))

        service = PyTorchInferenceService(_build_logger())

        result = service.detect_input_size(config, dataset)

        assert result == (3, 100, 100)

    def test_returns_none_when_unavailable(self) -> None:
        """サイズ情報が一切取得できない場合に None を返すこと."""
        config = _build_config(val_transform=Compose([ToTensor()]))
        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=0)

        service = PyTorchInferenceService(_build_logger())

        result = service.detect_input_size(config, dataset)

        assert result is None


class TestRunInference:
    """run_inference のテスト."""

    def test_returns_predictions_and_metrics(self) -> None:
        """推論結果とメトリクスが正しく返されること."""
        service = PyTorchInferenceService(_build_logger())

        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = (
            torch.tensor([0, 1, 0]),
            torch.tensor([0.9, 0.8, 0.95]),
            {
                "avg_time_per_image": 5.0,
                "total_samples": 3,
                "warmup_samples": 1,
            },
        )
        mock_loader = MagicMock()

        labels, scores, metrics, e2e_time = service.run_inference(
            mock_predictor, mock_loader
        )

        assert labels == [0, 1, 0]
        assert len(scores) == 3
        assert metrics["avg_time_per_image"] == 5.0
        assert e2e_time > 0


class TestAggregateAndExport:
    """aggregate_and_export のテスト."""

    @patch(
        "pochitrain.inference.services.pytorch_inference_service.ResultExportService"
    )
    @patch(
        "pochitrain.inference.services.pytorch_inference_service.log_inference_result"
    )
    def test_calls_log_and_export(
        self,
        mock_log_result: MagicMock,
        mock_export_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """log_inference_result と ResultExportService.export が呼ばれること."""
        service = PyTorchInferenceService(_build_logger())

        mock_dataset = MagicMock()
        mock_dataset.get_file_paths.return_value = ["a.jpg", "b.jpg"]
        mock_dataset.labels = [0, 1]
        mock_dataset.get_classes.return_value = ["cat", "dog"]

        service.aggregate_and_export(
            workspace_dir=tmp_path,
            model_path=Path("model.pth"),
            data_path=Path("data/val"),
            dataset=mock_dataset,
            predicted_labels=[0, 1],
            confidence_scores=[0.9, 0.8],
            metrics={
                "avg_time_per_image": 5.0,
                "total_samples": 2,
                "warmup_samples": 1,
            },
            e2e_total_time_ms=100.0,
            input_size=(3, 224, 224),
            model_info={"model_name": "resnet18"},
            cm_config=None,
        )

        mock_log_result.assert_called_once()
        mock_export_cls.return_value.export.assert_called_once()

        # export に渡された ResultExportRequest の内容を検証
        call_args = mock_export_cls.return_value.export.call_args
        request = call_args[0][0]
        assert request.num_samples == 2
        assert request.correct == 2  # [0,1] vs [0,1] -> 2 correct
        assert request.results_filename == "pytorch_inference_results.csv"
        assert request.summary_filename == "pytorch_inference_summary.txt"
