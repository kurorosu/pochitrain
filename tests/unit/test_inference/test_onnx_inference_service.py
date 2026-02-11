"""OnnxInferenceService のテスト."""

from pathlib import Path

import pytest

from pochitrain.inference.services.onnx_inference_service import OnnxInferenceService
from pochitrain.inference.types.orchestration_types import InferenceCliRequest


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
