"""TensorRTInferenceService のテスト."""

from pathlib import Path

import pytest

from pochitrain.inference.services.trt_inference_service import TensorRTInferenceService
from pochitrain.inference.types.orchestration_types import InferenceCliRequest


class TestResolvePipeline:
    """resolve_pipeline のテスト."""

    def test_auto_returns_gpu(self):
        """auto 指定時は gpu に解決される."""
        service = TensorRTInferenceService()
        assert service.resolve_pipeline("auto") == "gpu"

    def test_passthrough_for_non_auto(self):
        """auto 以外はそのまま返す."""
        service = TensorRTInferenceService()
        assert service.resolve_pipeline("fast") == "fast"


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
            config={"num_workers": 2, "pin_memory": False},
            pipeline="gpu",
        )

        assert options.batch_size == 1
        assert options.num_workers == 2
        assert options.pin_memory is False
        assert options.use_gpu is True
        assert options.use_gpu_pipeline is True


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
