"""TensorRTInferenceService のテスト."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from pochitrain.inference.adapters.trt_runtime_adapter import TensorRTRuntimeAdapter
from pochitrain.inference.services.trt_inference_service import TensorRTInferenceService
from pochitrain.inference.types.execution_types import ExecutionRequest, ExecutionResult
from pochitrain.inference.types.orchestration_types import (
    InferenceCliRequest,
    RuntimeExecutionRequest,
)


class TestResolvePipeline:
    """resolve_pipeline のテスト."""

    def test_auto_returns_gpu(self):
        """auto 指定時は gpu に解決される."""
        service = TensorRTInferenceService()
        assert service.resolve_pipeline("auto", use_gpu=True) == "gpu"

    def test_passthrough_for_non_auto(self):
        """auto 以外はそのまま返す."""
        service = TensorRTInferenceService()
        assert service.resolve_pipeline("fast", use_gpu=True) == "fast"


class TestResolvePaths:
    """resolve_paths のテスト."""

    def test_resolve_paths_with_explicit_data_and_output(self, tmp_path: Path):
        """--data, --output 指定時にその値を採用する."""
        data_path = tmp_path / "data"
        data_path.mkdir()
        output_dir = tmp_path / "out"

        request = InferenceCliRequest(
            model_path=tmp_path / "model.engine",
            data_path=data_path,
            output_dir=output_dir,
            requested_pipeline="auto",
        )
        resolved = TensorRTInferenceService().resolve_paths(request, config={})

        assert resolved.data_path == data_path
        assert resolved.output_dir == output_dir
        assert output_dir.exists()

    def test_resolve_paths_raises_when_data_is_unresolved(self, tmp_path: Path):
        """データパス未指定かつconfig未設定なら ValueError."""
        request = InferenceCliRequest(
            model_path=tmp_path / "model.engine",
            data_path=None,
            output_dir=tmp_path / "out",
            requested_pipeline="auto",
        )

        with pytest.raises(ValueError):
            TensorRTInferenceService().resolve_paths(request, config={})


class TestResolveRuntimeOptions:
    """resolve_runtime_options のテスト."""

    def test_runtime_options_use_tensorrt_defaults(self):
        """TensorRT実行向けの固定値を組み立てる."""
        service = TensorRTInferenceService()
        options = service.resolve_runtime_options(
            config={"num_workers": 2, "infer_pin_memory": False},
            pipeline="gpu",
        )

        assert options.batch_size == 1
        assert options.num_workers == 2
        assert options.pin_memory is False
        assert options.use_gpu is True
        assert options.use_gpu_pipeline is True


class TestInferenceAndAdapter:
    """TensorRT 推論インスタンスとアダプタ生成のテスト."""

    def test_create_trt_inference_uses_lazy_import(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """遅延 import した TensorRTInference を使ってインスタンス化する."""

        class _FakeTensorRTInference:
            def __init__(self, engine_path: Path) -> None:
                self.engine_path = engine_path

        monkeypatch.setattr(
            "pochitrain.tensorrt.TensorRTInference",
            _FakeTensorRTInference,
        )

        service = TensorRTInferenceService()
        inference = service.create_trt_inference(tmp_path / "model.engine")
        assert isinstance(inference, _FakeTensorRTInference)

    def test_resolve_val_transform_returns_config_value(self) -> None:
        """config に val_transform があればその値を返す."""
        service = TensorRTInferenceService()
        transform = object()
        resolved = service.resolve_val_transform(
            config={"val_transform": transform},
            inference=object(),
        )
        assert resolved is transform

    def test_resolve_val_transform_builds_from_engine_shape(self) -> None:
        """config 未指定時はエンジン入力サイズから transform を組み立てる."""

        class _DummyInference:
            def get_input_shape(self) -> tuple[int, int, int, int]:
                return (1, 3, 224, 224)

        service = TensorRTInferenceService()
        resolved = service.resolve_val_transform(config={}, inference=_DummyInference())
        assert hasattr(resolved, "transforms")

    def test_create_runtime_adapter_returns_trt_adapter(self) -> None:
        """TensorRT 推論インスタンスから TRT アダプタを生成できる."""

        class _DummyInference:
            pass

        adapter = TensorRTInferenceService().create_runtime_adapter(_DummyInference())
        assert isinstance(adapter, TensorRTRuntimeAdapter)


class TestResolveInputSize:
    """resolve_input_size のテスト."""

    def test_resolve_input_size_returns_tuple_when_shape_is_4d(self):
        """4次元shapeなら (C, H, W) を返す."""
        service = TensorRTInferenceService()
        assert service.resolve_input_size([1, 3, 224, 224]) == (3, 224, 224)

    def test_resolve_input_size_returns_none_when_shape_is_not_4d(self):
        """4次元でないshapeなら None を返す."""
        service = TensorRTInferenceService()
        assert service.resolve_input_size([1, 3, 224]) is None


class TestRun:
    """run のテスト."""

    def test_run_returns_inference_run_result(self) -> None:
        """ExecutionService の結果を共通結果型へ変換することを検証する."""
        service = TensorRTInferenceService()
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
