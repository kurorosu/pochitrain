"""OnnxInferenceService のテスト."""

from pathlib import Path

import pytest

from pochitrain.inference.adapters.onnx_runtime_adapter import OnnxRuntimeAdapter
from pochitrain.inference.services.onnx_inference_service import OnnxInferenceService

# Why:
# 共通ロジックは test_base_inference_service.py で検証済みのため、
# ここでは ONNX 固有差分のみを検証する.


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
