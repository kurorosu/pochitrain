"""ExecutionService の単体テスト."""

from typing import List

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from pochitrain.inference.execution_service import ExecutionService
from pochitrain.inference.execution_types import ExecutionRequest


class _DummyRuntimeAdapter:
    """ExecutionService テスト用のダミーランタイム."""

    def __init__(
        self,
        logits_sequence: List[np.ndarray],
        use_cuda_timing: bool = False,
    ) -> None:
        """ダミーランタイムを初期化する.

        Args:
            logits_sequence: get_output で返すロジット配列の順序.
            use_cuda_timing: CUDA Event 計測を使用可能かどうか.
        """
        self._logits_sequence = list(logits_sequence)
        self._use_cuda_timing = use_cuda_timing
        self.warmup_called = False
        self.warmup_repeats = 0
        self.set_input_calls = 0
        self.run_calls = 0

    @property
    def use_cuda_timing(self) -> bool:
        """CUDA Event 計測の可否を返す."""
        return self._use_cuda_timing

    def warmup(self, image: torch.Tensor, request: ExecutionRequest) -> None:
        """ウォームアップ呼び出しを記録する."""
        self.warmup_called = True
        self.warmup_repeats = request.warmup_repeats

    def set_input(self, images: torch.Tensor, request: ExecutionRequest) -> None:
        """入力設定呼び出しを記録する."""
        self.set_input_calls += 1

    def run_inference(self) -> None:
        """推論実行呼び出しを記録する."""
        self.run_calls += 1

    def get_output(self) -> np.ndarray:
        """ロジット配列を順番に返す."""
        return self._logits_sequence.pop(0)


def _build_loader(labels: List[int]) -> DataLoader:
    """テスト用 DataLoader を作成する.

    Args:
        labels: 正解ラベル列.

    Returns:
        バッチサイズ1のDataLoader.
    """
    images = torch.zeros((len(labels), 3, 8, 8), dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(images, label_tensor)
    return DataLoader(dataset, batch_size=1, shuffle=False)


def test_run_aggregates_predictions_and_timing(monkeypatch) -> None:
    """ウォームアップ除外と計測集計が期待どおりに動作する."""
    loader = _build_loader([1, 0, 1])
    runtime = _DummyRuntimeAdapter(
        logits_sequence=[
            np.array([[0.1, 0.9]], dtype=np.float32),
            np.array([[0.8, 0.2]], dtype=np.float32),
            np.array([[0.2, 0.8]], dtype=np.float32),
        ]
    )
    request = ExecutionRequest(
        use_gpu_pipeline=False,
        warmup_repeats=10,
        skip_measurement_batches=1,
        use_cuda_timing=False,
    )

    perf_counter_values = iter([0.0, 1.0, 1.003, 2.0, 2.004, 3.0])
    monkeypatch.setattr(
        "pochitrain.inference.execution_service.time.perf_counter",
        lambda: next(perf_counter_values),
    )

    result = ExecutionService().run(loader, runtime, request)

    assert runtime.warmup_called is True
    assert runtime.warmup_repeats == 10
    assert runtime.set_input_calls == 3
    assert runtime.run_calls == 3

    assert result.predictions == [1, 0, 1]
    assert result.true_labels == [1, 0, 1]
    assert len(result.confidences) == 3
    assert result.total_samples == 2
    assert result.warmup_samples == 1
    assert result.total_inference_time_ms == pytest.approx(7.0)
    assert result.e2e_total_time_ms == pytest.approx(3000.0)


def test_run_skips_warmup_when_repeats_is_zero(monkeypatch) -> None:
    """warmup_repeats=0 の場合はウォームアップを実行しない."""
    loader = _build_loader([0, 1])
    runtime = _DummyRuntimeAdapter(
        logits_sequence=[
            np.array([[0.9, 0.1]], dtype=np.float32),
            np.array([[0.1, 0.9]], dtype=np.float32),
        ]
    )
    request = ExecutionRequest(
        use_gpu_pipeline=False,
        warmup_repeats=0,
        skip_measurement_batches=0,
        use_cuda_timing=False,
    )

    perf_counter_values = iter([0.0, 1.0, 1.002, 2.0, 2.003, 4.0])
    monkeypatch.setattr(
        "pochitrain.inference.execution_service.time.perf_counter",
        lambda: next(perf_counter_values),
    )

    result = ExecutionService().run(loader, runtime, request)

    assert runtime.warmup_called is False
    assert result.warmup_samples == 0
    assert result.total_samples == 2
    assert result.predictions == [0, 1]
    assert result.total_inference_time_ms == pytest.approx(5.0)
    assert result.e2e_total_time_ms == pytest.approx(4000.0)


def test_run_uses_cuda_event_timing_when_available(monkeypatch) -> None:
    """CUDA Event 計測有効時は elapsed_time を合算して返す."""
    loader = _build_loader([0, 1])
    runtime = _DummyRuntimeAdapter(
        logits_sequence=[
            np.array([[0.9, 0.1]], dtype=np.float32),
            np.array([[0.2, 0.8]], dtype=np.float32),
        ],
        use_cuda_timing=True,
    )
    request = ExecutionRequest(
        use_gpu_pipeline=True,
        warmup_repeats=1,
        skip_measurement_batches=1,
        use_cuda_timing=True,
    )

    class _FakeCudaEvent:
        def __init__(self, enable_timing: bool = True) -> None:
            self.enable_timing = enable_timing

        def record(self) -> None:
            return None

        def elapsed_time(self, other: object) -> float:
            return 12.5

    perf_counter_values = iter([10.0, 10.8])
    monkeypatch.setattr(
        "pochitrain.inference.execution_service.time.perf_counter",
        lambda: next(perf_counter_values),
    )
    monkeypatch.setattr(
        "pochitrain.inference.execution_service.torch.cuda.Event",
        _FakeCudaEvent,
    )
    monkeypatch.setattr(
        "pochitrain.inference.execution_service.torch.cuda.synchronize",
        lambda: None,
    )

    result = ExecutionService().run(loader, runtime, request)

    assert runtime.warmup_called is True
    assert result.warmup_samples == 1
    assert result.total_samples == 1
    assert result.total_inference_time_ms == pytest.approx(12.5)
    assert result.e2e_total_time_ms == pytest.approx(800.0)
