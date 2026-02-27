"""ONNX/TRT RuntimeAdapter の単体テスト."""

from typing import Any, cast

import torch

from pochitrain.inference.adapters.onnx_runtime_adapter import OnnxRuntimeAdapter
from pochitrain.inference.adapters.pytorch_runtime_adapter import PyTorchRuntimeAdapter
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
        self.sync_called = 0

    def set_input_gpu(self, tensor: torch.Tensor) -> None:
        self.last_gpu_input = tensor

    def set_input(self, image: Any) -> None:
        return None

    def run_pure(self) -> None:
        return None

    def get_output(self) -> Any:
        return None

    def synchronize_input_if_needed(self) -> None:
        self.sync_called += 1


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
        "pochitrain.inference.adapters.engine_runtime_adapter.gpu_normalize",
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
        "pochitrain.inference.adapters.engine_runtime_adapter.gpu_normalize",
        _fake_gpu_normalize,
    )

    adapter = OnnxRuntimeAdapter(cast(Any, _StubOnnxInference()))
    inference = _StubOnnxInference()
    request = ExecutionRequest(
        use_gpu_pipeline=True,
        mean_255=torch.ones((1, 3, 1, 1), dtype=torch.float32),
        std_255=torch.ones((1, 3, 1, 1), dtype=torch.float32),
        gpu_non_blocking=False,
    )
    images = torch.randint(0, 256, (1, 3, 4, 4), dtype=torch.uint8)

    adapter = OnnxRuntimeAdapter(cast(Any, inference))
    adapter.set_input(images, request)

    assert captured["non_blocking"] is False
    assert inference.sync_called == 0


def test_onnx_adapter_syncs_when_gpu_non_blocking_true(monkeypatch) -> None:
    """ONNX adapter が non_blocking=True の時に同期フックを呼ぶことを確認."""
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
        "pochitrain.inference.adapters.engine_runtime_adapter.gpu_normalize",
        _fake_gpu_normalize,
    )

    inference = _StubOnnxInference()
    adapter = OnnxRuntimeAdapter(cast(Any, inference))
    request = ExecutionRequest(
        use_gpu_pipeline=True,
        mean_255=torch.ones((1, 3, 1, 1), dtype=torch.float32),
        std_255=torch.ones((1, 3, 1, 1), dtype=torch.float32),
        gpu_non_blocking=True,
    )
    images = torch.randint(0, 256, (1, 3, 4, 4), dtype=torch.uint8)

    adapter.set_input(images, request)

    assert captured["non_blocking"] is True
    assert inference.sync_called == 1


def test_pytorch_adapter_runs_model_and_returns_numpy() -> None:
    """PyTorch adapter がモデル実行結果を numpy で返すことを確認."""

    class _StubModel:
        def __init__(self) -> None:
            self.calls = 0

        def eval(self) -> None:
            return None

        def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return torch.ones((inputs.size(0), 2), dtype=torch.float32)

    class _StubPredictor:
        def __init__(self) -> None:
            self.device = torch.device("cpu")
            self.model = _StubModel()

    adapter = PyTorchRuntimeAdapter(cast(Any, _StubPredictor()))
    request = ExecutionRequest(use_gpu_pipeline=False)
    images = torch.zeros((1, 3, 4, 4), dtype=torch.float32)

    adapter.set_input(images, request)
    adapter.run_inference()
    output = adapter.get_output()

    assert output.shape == (1, 2)
    assert adapter.predictor.model.calls == 1
