"""orchestration_types のユニットテスト."""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from pochitrain.inference.interfaces import IRuntimeAdapter
from pochitrain.inference.types.execution_types import ExecutionRequest, ExecutionResult
from pochitrain.inference.types.orchestration_types import (
    InferenceRunResult,
    PyTorchRunRequest,
    RuntimeExecutionRequest,
)


class _DummyRuntimeAdapter(IRuntimeAdapter):
    @property
    def use_cuda_timing(self) -> bool:
        return False

    def get_timing_stream(self) -> torch.cuda.Stream | None:
        return None

    def warmup(self, image: torch.Tensor, request: ExecutionRequest) -> None:
        return None

    def set_input(self, images: torch.Tensor, request: ExecutionRequest) -> None:
        return None

    def run_inference(self) -> None:
        return None

    def get_output(self) -> np.ndarray:
        return np.zeros((1, 2), dtype=np.float32)


def test_inference_run_result_from_execution_result() -> None:
    """ExecutionResult から共通結果型へ変換できることを検証する."""
    execution_result = ExecutionResult(
        predictions=[1, 0, 1],
        confidences=[0.9, 0.8, 0.7],
        true_labels=[1, 0, 0],
        total_inference_time_ms=12.0,
        total_samples=3,
        warmup_samples=1,
        e2e_total_time_ms=30.0,
    )

    result = InferenceRunResult.from_execution_result(execution_result)

    assert result.num_samples == 3
    assert result.correct == 2
    assert result.avg_time_per_image == 4.0
    assert result.avg_total_time_per_image == 10.0
    assert result.accuracy_percent == (2 / 3) * 100.0


def test_inference_run_result_from_components_with_zero_samples() -> None:
    """測定サンプル数が0でもゼロ除算せず集計できることを検証する."""
    result = InferenceRunResult.from_components(
        predictions=[],
        confidences=[],
        true_labels=[],
        total_inference_time_ms=0.0,
        total_samples=0,
        warmup_samples=0,
        e2e_total_time_ms=0.0,
    )

    assert result.num_samples == 0
    assert result.correct == 0
    assert result.avg_time_per_image == 0.0
    assert result.avg_total_time_per_image == 0.0
    assert result.accuracy_percent == 0.0


def test_runtime_execution_request_holds_common_context() -> None:
    """RuntimeExecutionRequest が実行コンテキストを保持できることを検証する."""
    loader = DataLoader(
        TensorDataset(torch.zeros((1, 3, 32, 32)), torch.tensor([0])),
        batch_size=1,
    )
    request = RuntimeExecutionRequest(
        data_loader=loader,
        runtime_adapter=_DummyRuntimeAdapter(),
        execution_request=ExecutionRequest(use_gpu_pipeline=False),
    )

    assert request.data_loader is loader
    assert request.execution_request.use_gpu_pipeline is False


def test_pytorch_run_request_holds_predictor_and_loader() -> None:
    """PyTorchRunRequest が predictor と loader を保持できることを検証する."""
    loader = DataLoader(
        TensorDataset(torch.zeros((1, 3, 32, 32)), torch.tensor([0])),
        batch_size=1,
    )
    predictor = object()
    request = PyTorchRunRequest(predictor=predictor, val_loader=loader)

    assert request.predictor is predictor
    assert request.val_loader is loader
