"""TensorRTInferenceService のテスト."""

from pathlib import Path

import pytest

from pochitrain.inference.adapters.trt_runtime_adapter import TensorRTRuntimeAdapter
from pochitrain.inference.services.trt_inference_service import TensorRTInferenceService

# Why:
# 共通ロジックは test_base_inference_service.py で検証済みのため,
# ここでは TensorRT 固有差分のみを検証する.


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
