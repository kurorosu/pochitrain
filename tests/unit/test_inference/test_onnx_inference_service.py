"""OnnxInferenceService の runtime 固有差分テスト."""

from pathlib import Path

import pytest

from pochitrain.inference.services.onnx_inference_service import OnnxInferenceService

# Why:
# 共通ロジックは test_base_inference_service.py で検証済みのため、
# ここでは ONNX 固有差分のみを検証する.


class TestOnnxSpecificBehavior:
    """ONNX 固有差分のテスト."""

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

    @pytest.mark.parametrize(
        ("shape", "expected"),
        [
            ([1, 3, 224, 224], (3, 224, 224)),
            ([1, 3, "h", "w"], None),
            ([1, 3, 224], None),
        ],
    )
    def test_resolve_input_size(self, shape: list[object], expected: object) -> None:
        """ONNX の入力shape解決ルールを検証する."""
        assert OnnxInferenceService().resolve_input_size(shape) == expected
