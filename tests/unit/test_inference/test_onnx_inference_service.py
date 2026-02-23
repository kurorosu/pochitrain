"""OnnxInferenceService のテスト."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from pochitrain.inference.adapters.onnx_runtime_adapter import OnnxRuntimeAdapter
from pochitrain.inference.services.onnx_inference_service import OnnxInferenceService
from pochitrain.inference.types.execution_types import ExecutionRequest, ExecutionResult
from pochitrain.inference.types.orchestration_types import (
    InferenceCliRequest,
    RuntimeExecutionRequest,
)


class TestResolvePipeline:
    """resolve_pipeline のテスト."""

    def test_auto_with_gpu_returns_gpu(self):
        """auto + GPU推論なら gpu を返す."""
        service = OnnxInferenceService()
        assert service.resolve_pipeline("auto", use_gpu=True) == "gpu"

    def test_auto_without_gpu_returns_fast(self):
        """auto + CPU推論なら fast を返す."""
        service = OnnxInferenceService()
        assert service.resolve_pipeline("auto", use_gpu=False) == "fast"

    def test_gpu_without_gpu_falls_back_to_fast(self):
        """gpu指定でもGPU不可なら fast にフォールバックする."""
        service = OnnxInferenceService()
        assert service.resolve_pipeline("gpu", use_gpu=False) == "fast"


class TestResolvePaths:
    """resolve_paths のテスト."""

    def test_resolve_paths_with_explicit_data_and_output(self, tmp_path: Path):
        """--data, --output 指定時にその値を採用する."""
        data_path = tmp_path / "data"
        data_path.mkdir()
        output_dir = tmp_path / "out"

        request = InferenceCliRequest(
            model_path=tmp_path / "model.onnx",
            data_path=data_path,
            output_dir=output_dir,
            requested_pipeline="auto",
        )
        resolved = OnnxInferenceService().resolve_paths(request, config={})

        assert resolved.data_path == data_path
        assert resolved.output_dir == output_dir
        assert output_dir.exists()

    def test_resolve_paths_raises_when_data_is_unresolved(self, tmp_path: Path):
        """データパス未指定かつconfig未設定なら ValueError."""
        request = InferenceCliRequest(
            model_path=tmp_path / "model.onnx",
            data_path=None,
            output_dir=tmp_path / "out",
            requested_pipeline="auto",
        )

        with pytest.raises(ValueError):
            OnnxInferenceService().resolve_paths(request, config={})


class TestResolveRuntimeOptions:
    """resolve_runtime_options のテスト."""

    def test_runtime_options_from_config(self):
        """設定値から実行オプションを組み立てる."""
        service = OnnxInferenceService()
        options = service.resolve_runtime_options(
            config={"batch_size": 8, "num_workers": 4, "pin_memory": False},
            pipeline="gpu",
            use_gpu=True,
        )

        assert options.batch_size == 8
        assert options.num_workers == 4
        assert options.pin_memory is False
        assert options.use_gpu is True
        assert options.use_gpu_pipeline is True


class TestSessionAndAdapter:
    """ONNX セッション作成とアダプタ生成のテスト."""

    def test_create_onnx_session_returns_actual_use_gpu_flag(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """内部フォールバック時に actual_use_gpu=False を返す."""

        class _FakeOnnxInference:
            def __init__(self, model_path: Path, use_gpu: bool = False) -> None:
                self.model_path = model_path
                self.use_gpu = False

        monkeypatch.setattr(
            "pochitrain.onnx.OnnxInference",
            _FakeOnnxInference,
        )

        service = OnnxInferenceService()
        inference, actual_use_gpu = service.create_onnx_session(
            tmp_path / "model.onnx",
            use_gpu=True,
        )

        assert isinstance(inference, _FakeOnnxInference)
        assert actual_use_gpu is False

    def test_create_runtime_adapter_returns_onnx_adapter(self) -> None:
        """ONNX 推論インスタンスから ONNX アダプタを生成できる."""

        class _DummyInference:
            use_gpu = False

            def run_pure(self) -> None:
                return None

        adapter = OnnxInferenceService().create_runtime_adapter(_DummyInference())
        assert isinstance(adapter, OnnxRuntimeAdapter)


class TestResolveInputSize:
    """resolve_input_size のテスト."""

    def test_resolve_input_size_returns_tuple_when_shape_is_static(self):
        """静的shapeなら (C, H, W) を返す."""
        service = OnnxInferenceService()
        assert service.resolve_input_size([1, 3, 224, 224]) == (3, 224, 224)

    def test_resolve_input_size_returns_none_when_shape_is_dynamic(self):
        """空間次元が動的なら None を返す."""
        service = OnnxInferenceService()
        assert service.resolve_input_size([1, 3, "h", "w"]) is None


class TestRun:
    """run のテスト."""

    def test_run_returns_inference_run_result(self) -> None:
        """ExecutionService の結果を共通結果型へ変換することを検証する."""
        service = OnnxInferenceService()
        data_loader = DataLoader(
            TensorDataset(torch.zeros((1, 3, 32, 32)), torch.tensor([0])),
            batch_size=1,
        )

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

            def get_output(self) -> np.ndarray:
                return np.zeros((1, 2), dtype=np.float32)

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
