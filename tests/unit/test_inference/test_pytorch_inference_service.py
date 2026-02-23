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
from pochitrain.inference.types.orchestration_types import (
    InferenceCliRequest,
    InferenceRunResult,
    PyTorchRunRequest,
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


class TestResolvePipeline:
    """resolve_pipeline のテスト."""

    def test_auto_returns_current(self) -> None:
        """auto 指定時は current を返すこと."""
        service = PyTorchInferenceService(_build_logger())
        assert service.resolve_pipeline("auto") == "current"

    def test_non_auto_also_returns_current(self) -> None:
        """非auto指定時も current を返すこと."""
        service = PyTorchInferenceService(_build_logger())
        assert service.resolve_pipeline("gpu") == "current"


class TestResolvePaths:
    """resolve_paths のテスト."""

    def test_resolve_paths_with_explicit_data_and_output(self, tmp_path: Path) -> None:
        """--data, --output 指定時にその値を使うこと."""
        data_path = tmp_path / "data"
        data_path.mkdir()
        output_dir = tmp_path / "out"

        request = InferenceCliRequest(
            model_path=tmp_path / "model.pth",
            data_path=data_path,
            output_dir=output_dir,
            requested_pipeline="auto",
        )
        resolved = PyTorchInferenceService(_build_logger()).resolve_paths(
            request, config={}
        )

        assert resolved.data_path == data_path
        assert resolved.output_dir == output_dir
        assert output_dir.exists()

    def test_resolve_paths_raises_when_data_is_unresolved(self, tmp_path: Path) -> None:
        """データパスを解決できない場合は ValueError を送出すること."""
        request = InferenceCliRequest(
            model_path=tmp_path / "model.pth",
            data_path=None,
            output_dir=tmp_path / "out",
            requested_pipeline="auto",
        )

        with pytest.raises(ValueError):
            PyTorchInferenceService(_build_logger()).resolve_paths(request, config={})


class TestResolveRuntimeOptions:
    """resolve_runtime_options のテスト."""

    def test_runtime_options_from_config(self) -> None:
        """設定値から実行オプションを解決できること."""
        service = PyTorchInferenceService(_build_logger())
        options = service.resolve_runtime_options(
            config={"batch_size": 8, "num_workers": 4, "pin_memory": False},
            pipeline="current",
            use_gpu=False,
        )

        assert options.pipeline == "current"
        assert options.batch_size == 8
        assert options.num_workers == 4
        assert options.pin_memory is False
        assert options.use_gpu is False
        assert options.use_gpu_pipeline is False


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
        mock_loader.dataset = MagicMock(labels=[0, 1, 0])

        result = service.run_inference(mock_predictor, mock_loader)

        assert result.predictions == [0, 1, 0]
        assert len(result.confidences) == 3
        assert result.avg_time_per_image == 5.0
        assert result.num_samples == 3
        assert result.correct == 3
        assert result.avg_total_time_per_image > 0


class TestRun:
    """run のテスト."""

    def test_run_delegates_to_run_inference(self) -> None:
        """run が run_inference を委譲呼び出しすることを検証する."""
        service = PyTorchInferenceService(_build_logger())

        expected = InferenceRunResult(
            predictions=[0],
            confidences=[0.9],
            true_labels=[0],
            num_samples=1,
            correct=1,
            avg_time_per_image=1.0,
            total_samples=1,
            warmup_samples=0,
            avg_total_time_per_image=2.0,
        )

        with patch.object(
            service,
            "run_inference",
            return_value=expected,
        ) as mock_run_inference:
            request = PyTorchRunRequest(
                predictor=MagicMock(),
                val_loader=MagicMock(),
            )
            result = service.run(request)

        mock_run_inference.assert_called_once_with(
            predictor=request.predictor,
            val_loader=request.val_loader,
        )
        assert result == expected


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
            run_result=InferenceRunResult(
                predictions=[0, 1],
                confidences=[0.9, 0.8],
                true_labels=[0, 1],
                num_samples=2,
                correct=2,
                avg_time_per_image=5.0,
                total_samples=2,
                warmup_samples=1,
                avg_total_time_per_image=50.0,
            ),
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
