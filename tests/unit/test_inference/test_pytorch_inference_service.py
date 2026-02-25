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
from pochitrain.inference.types.execution_types import ExecutionRequest, ExecutionResult
from pochitrain.inference.types.orchestration_types import (
    InferenceCliRequest,
    InferenceRunResult,
    InferenceRuntimeOptions,
    RuntimeExecutionRequest,
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


class TestCreateDataloader:
    """create_dataloader のテスト."""

    @patch(
        "pochitrain.inference.services.pytorch_inference_service.create_dataset_and_params"
    )
    def test_returns_loader_and_dataset(self, mock_create_dataset: MagicMock) -> None:
        """DataLoader とデータセットが返されること."""
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
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
            use_gpu=False,
            use_gpu_pipeline=True,
        )

        loader, dataset, pipeline, norm_mean, norm_std = service.create_dataloader(
            {},
            data_path,
            config.val_transform,
            "gpu",
            runtime_options,
        )

        mock_create_dataset.assert_called_once_with(
            "gpu",
            data_path,
            config.val_transform,
        )
        assert dataset is mock_dataset
        assert loader.batch_size == config.batch_size
        assert pipeline == "fast"
        assert norm_mean == [0.485, 0.456, 0.406]
        assert norm_std == [0.229, 0.224, 0.225]


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


class TestRun:
    """run のテスト."""

    def test_run_returns_inference_run_result(self) -> None:
        """ExecutionService の結果を共通結果型へ変換することを検証する."""
        service = PyTorchInferenceService(_build_logger())
        data_loader = MagicMock()

        class _DummyRuntimeAdapter:
            @property
            def use_cuda_timing(self) -> bool:
                return False

            def get_timing_stream(self) -> torch.cuda.Stream | None:
                return None

            def warmup(self, image: torch.Tensor, request: ExecutionRequest) -> None:
                return None

            def set_input(
                self, images: torch.Tensor, request: ExecutionRequest
            ) -> None:
                return None

            def run_inference(self) -> None:
                return None

            def get_output(self):
                return None

        class _FakeExecutionService:
            def run(self, data_loader, runtime, request):  # noqa: ANN001
                return ExecutionResult(
                    predictions=[1, 0],
                    confidences=[0.9, 0.8],
                    true_labels=[1, 1],
                    total_inference_time_ms=6.0,
                    total_samples=2,
                    warmup_samples=1,
                    e2e_total_time_ms=12.0,
                )

        result = service.run(
            RuntimeExecutionRequest(
                data_loader=data_loader,
                runtime_adapter=_DummyRuntimeAdapter(),
                execution_request=ExecutionRequest(use_gpu_pipeline=False),
            ),
            execution_service=_FakeExecutionService(),
        )

        assert result.correct == 1
        assert result.num_samples == 2
        assert result.avg_time_per_image == 3.0
        assert result.avg_total_time_per_image == 6.0


class TestAggregateAndExport:
    """aggregate_and_export のテスト."""

    @patch("pochitrain.inference.services.interfaces.ResultExportService")
    @patch("pochitrain.inference.services.interfaces.log_inference_result")
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
            results_filename="pytorch_inference_results.csv",
            summary_filename="pytorch_inference_summary.txt",
        )

        mock_log_result.assert_called_once()
        mock_export_cls.return_value.export.assert_called_once()

        call_args = mock_export_cls.return_value.export.call_args
        request = call_args[0][0]
        assert request.num_samples == 2
        assert request.correct == 2  # [0,1] vs [0,1] -> 2 correct
        assert request.results_filename == "pytorch_inference_results.csv"
        assert request.summary_filename == "pytorch_inference_summary.txt"
