"""TensorRTInferenceService の runtime 固有差分テスト."""

import pytest

from pochitrain.inference.services.trt_inference_service import TensorRTInferenceService

# Why:
# 共通ロジックは test_base_inference_service.py で検証済みのため,
# ここでは TensorRT 固有差分のみを検証する.


class TestTrtSpecificBehavior:
    """TensorRT 固有差分のテスト."""

    def test_resolve_pipeline_and_runtime_options_use_trt_defaults(self) -> None:
        """auto→gpu 解決と batch_size=1 の固定仕様を検証する."""
        service = TensorRTInferenceService()
        pipeline = service.resolve_pipeline("auto", use_gpu=False)
        options = service.resolve_runtime_options(
            config={"num_workers": 2, "infer_pin_memory": False},
            pipeline=pipeline,
        )

        assert pipeline == "gpu"
        assert options.batch_size == 1
        assert options.num_workers == 2
        assert options.pin_memory is False
        assert options.use_gpu is True
        assert options.use_gpu_pipeline is True

    def test_resolve_val_transform_prefers_config_then_falls_back(self) -> None:
        """val_transform は設定優先, 未指定時はエンジンshapeから生成される."""

        class _DummyInference:
            def get_input_shape(self) -> tuple[int, int, int, int]:
                return (1, 3, 224, 224)

        service = TensorRTInferenceService()
        configured_transform = object()
        resolved_from_config = service.resolve_val_transform(
            config={"val_transform": configured_transform},
            inference=object(),
        )
        resolved_from_engine = service.resolve_val_transform(
            config={},
            inference=_DummyInference(),
        )

        assert resolved_from_config is configured_transform
        assert hasattr(resolved_from_engine, "transforms")
