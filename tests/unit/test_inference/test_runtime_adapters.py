"""ONNX/TRT RuntimeAdapter の単体テスト."""

from typing import Any, cast

import torch

from pochitrain.inference.adapters.onnx_runtime_adapter import OnnxRuntimeAdapter
from pochitrain.inference.adapters.trt_runtime_adapter import TensorRTRuntimeAdapter
from pochitrain.inference.types.execution_types import ExecutionRequest


class _StubTrtInference:
    """TensorRTRuntimeAdapter テスト用の最小スタブ."""

    def __init__(self) -> None:
        self.last_gpu_input: torch.Tensor | None = None

    @property
    def stream(self) -> torch.cuda.Stream:
        return cast(torch.cuda.Stream, object())

    def set_input_gpu(self, tensor: torch.Tensor) -> None:
        self.last_gpu_input = tensor

    def set_input(self, image: Any) -> None:
        return None

    def execute(self) -> None:
        return None

    def get_output(self) -> Any:
        return None


class _StubOnnxInference:
    """OnnxRuntimeAdapter テスト用の最小スタブ."""

    def __init__(self) -> None:
        self.use_gpu = True
        self.last_gpu_input: torch.Tensor | None = None

    def set_input_gpu(self, tensor: torch.Tensor) -> None:
        self.last_gpu_input = tensor

    def set_input(self, image: Any) -> None:
        return None

    def run_pure(self) -> None:
        return None

    def get_output(self) -> Any:
        return None


def test_trt_adapter_passes_gpu_non_blocking_to_gpu_normalize(monkeypatch) -> None:
    """TRT adapter が gpu_non_blocking を gpu_normalize へ渡すことを確認."""
    captured: dict[str, Any] = {}

    def _fake_gpu_normalize(
        images: torch.Tensor,
        mean_255: torch.Tensor,
        std_255: torch.Tensor,
        non_blocking: bool = True,
    ) -> torch.Tensor:
        captured["non_blocking"] = non_blocking
        return images.to(dtype=torch.float32)

    monkeypatch.setattr(
        "pochitrain.inference.adapters.trt_runtime_adapter.gpu_normalize",
        _fake_gpu_normalize,
    )

    adapter = TensorRTRuntimeAdapter(cast(Any, _StubTrtInference()))
    request = ExecutionRequest(
        use_gpu_pipeline=True,
        mean_255=torch.ones((1, 3, 1, 1), dtype=torch.float32),
        std_255=torch.ones((1, 3, 1, 1), dtype=torch.float32),
        gpu_non_blocking=False,
    )
    images = torch.randint(0, 256, (1, 3, 4, 4), dtype=torch.uint8)

    adapter.set_input(images, request)

    assert captured["non_blocking"] is False


def test_onnx_adapter_passes_gpu_non_blocking_to_gpu_normalize(monkeypatch) -> None:
    """ONNX adapter が gpu_non_blocking を gpu_normalize へ渡すことを確認."""
    captured: dict[str, Any] = {}

    def _fake_gpu_normalize(
        images: torch.Tensor,
        mean_255: torch.Tensor,
        std_255: torch.Tensor,
        non_blocking: bool = True,
    ) -> torch.Tensor:
        captured["non_blocking"] = non_blocking
        return images.to(dtype=torch.float32)

    monkeypatch.setattr(
        "pochitrain.inference.adapters.onnx_runtime_adapter.gpu_normalize",
        _fake_gpu_normalize,
    )

    adapter = OnnxRuntimeAdapter(cast(Any, _StubOnnxInference()))
    request = ExecutionRequest(
        use_gpu_pipeline=True,
        mean_255=torch.ones((1, 3, 1, 1), dtype=torch.float32),
        std_255=torch.ones((1, 3, 1, 1), dtype=torch.float32),
        gpu_non_blocking=False,
    )
    images = torch.randint(0, 256, (1, 3, 4, 4), dtype=torch.uint8)

    adapter.set_input(images, request)

    assert captured["non_blocking"] is False
