"""infer_trt CLIの単体テスト."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest

from pochitrain.cli.infer_trt import PIPELINE_CHOICES, main
from pochitrain.inference.types.execution_types import ExecutionRequest
from pochitrain.inference.types.orchestration_types import (
    InferenceResolvedPaths,
    InferenceRunResult,
)


class _FakeInference:
    input_shape = (1, 3, 32, 32)

    @staticmethod
    def get_input_shape() -> tuple[int, int, int, int]:
        return _FakeInference.input_shape


class _FakeDataset:
    labels = [0, 1]

    @staticmethod
    def get_file_paths() -> list[str]:
        return ["a.jpg", "b.jpg"]

    @staticmethod
    def get_classes() -> list[str]:
        return ["cat", "dog"]


class _FakeTensorRTService:
    def __init__(self) -> None:
        self.called_workspace: Path | None = None
        self.called_data: Path | None = None

    @staticmethod
    def create_trt_inference(engine_path: Path) -> _FakeInference:
        _ = engine_path
        return _FakeInference()

    @staticmethod
    def resolve_paths(
        request: Any,
        config: dict[str, object],
    ) -> InferenceResolvedPaths:
        _ = config
        return InferenceResolvedPaths(
            model_path=request.model_path,
            data_path=request.data_path,
            output_dir=request.output_dir,
        )

    @staticmethod
    def resolve_val_transform(config: dict[str, object], inference: object) -> object:
        _ = config
        _ = inference
        return object()

    @staticmethod
    def resolve_pipeline(requested: str, use_gpu: bool) -> str:
        _ = use_gpu
        return requested

    @staticmethod
    def resolve_runtime_options(config: dict[str, object], pipeline: str) -> object:
        _ = config
        return SimpleNamespace(
            batch_size=1,
            pin_memory=False,
            num_workers=0,
            use_gpu_pipeline=(pipeline == "gpu"),
        )

    @staticmethod
    def create_dataloader(
        *,
        config: dict[str, object],
        data_path: Path,
        val_transform: object,
        pipeline: str,
        runtime_options: object,
    ) -> tuple[object, _FakeDataset, str, None, None]:
        _ = config
        _ = data_path
        _ = val_transform
        _ = runtime_options
        return object(), _FakeDataset(), pipeline, None, None

    @staticmethod
    def resolve_input_size(shape: tuple[int, int, int, int]) -> tuple[int, int, int]:
        return (shape[1], shape[2], shape[3])

    @staticmethod
    def create_runtime_adapter(inference: object) -> object:
        _ = inference
        return SimpleNamespace(device="cuda")

    @staticmethod
    def build_runtime_execution_request(
        *,
        data_loader: object,
        runtime_adapter: object,
        use_gpu_pipeline: bool,
        norm_mean: object,
        norm_std: object,
        use_cuda_timing: bool,
        gpu_non_blocking: bool,
    ) -> Any:
        _ = norm_mean
        _ = norm_std
        _ = use_cuda_timing
        execution_request = ExecutionRequest(
            use_gpu_pipeline=use_gpu_pipeline,
            gpu_non_blocking=gpu_non_blocking,
        )
        return SimpleNamespace(
            data_loader=data_loader,
            runtime_adapter=runtime_adapter,
            execution_request=execution_request,
        )

    @staticmethod
    def run(request: Any) -> InferenceRunResult:
        _ = request
        return InferenceRunResult(
            predictions=[0, 1],
            confidences=[0.9, 0.8],
            true_labels=[0, 1],
            num_samples=2,
            correct=2,
            avg_time_per_image=1.2,
            total_samples=2,
            warmup_samples=0,
            avg_total_time_per_image=1.6,
        )

    def aggregate_and_export(self, **kwargs: object) -> None:
        self.called_workspace = cast(Path, kwargs["workspace_dir"])
        self.called_data = cast(Path, kwargs["data_path"])


def test_pipeline_choices_are_expected() -> None:
    """公開されるパイプライン候補が期待どおりであることを確認する."""
    assert PIPELINE_CHOICES == ("auto", "current", "fast", "gpu")


def test_main_no_args_exits() -> None:
    """引数なしでは argparse により SystemExit する."""
    with patch("sys.argv", ["infer-trt"]):
        with pytest.raises(SystemExit):
            main()


def test_main_nonexistent_engine_exits(tmp_path: Path) -> None:
    """存在しないエンジンパス指定で SystemExit する."""
    fake_engine = str(tmp_path / "nonexistent.engine")
    with patch("sys.argv", ["infer-trt", fake_engine]):
        with pytest.raises(SystemExit):
            main()


def test_main_delegates_service_and_writes_benchmark_json(tmp_path: Path) -> None:
    """正常系でServiceに委譲し、benchmark_result.jsonを出力する."""
    import pochitrain.cli.infer_trt as infer_trt_module

    fake_service = _FakeTensorRTService()
    engine_path = tmp_path / "model_fp32.engine"
    data_path = tmp_path / "data"
    output_path = tmp_path / "output"
    engine_path.touch()
    data_path.mkdir()

    config = {
        "model_name": "resnet18",
        "gpu_non_blocking": True,
        "confusion_matrix_config": {},
    }

    with (
        patch(
            "sys.argv",
            [
                "infer-trt",
                str(engine_path),
                "--data",
                str(data_path),
                "--output",
                str(output_path),
                "--pipeline",
                "gpu",
                "--benchmark-json",
                "--benchmark-env-name",
                "TestEnv",
            ],
        ),
        patch.object(
            infer_trt_module, "TensorRTInferenceService", lambda: fake_service
        ),
        patch.object(infer_trt_module, "load_config_auto", lambda _path: config),
    ):
        main()

    benchmark_json = output_path / "benchmark_result.json"
    assert benchmark_json.exists()

    with benchmark_json.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    assert payload["runtime"] == "tensorrt"
    assert payload["pipeline"] == "gpu"
    assert payload["env_name"] == "TestEnv"
    assert fake_service.called_workspace == output_path
    assert fake_service.called_data == data_path
