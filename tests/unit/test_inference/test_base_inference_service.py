"""IInferenceService の基底クラスロジックのテスト.

Why:
    共通ロジックの検証はこのファイルに集約する.
    runtime 固有差分の検証は ONNX/TRT/PyTorch の各テストへ分離し,
    集約しすぎによる差分見落としを防ぐ.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import pytest
from torch.utils.data import DataLoader

from pochitrain.inference.services.interfaces import IInferenceService
from pochitrain.inference.types.orchestration_types import (
    InferenceCliRequest,
)


class ConcreteInferenceService(IInferenceService):
    """テスト用の具象クラス."""

    def resolve_input_size(self, shape: Any) -> Optional[tuple[int, int, int]]:
        return (3, 224, 224)


@pytest.fixture
def service():
    return ConcreteInferenceService(logger=logging.getLogger("test"))


class TestIInferenceServiceResolvePipeline:
    """resolve_pipeline のテスト."""

    @pytest.mark.parametrize(
        "requested, use_gpu, expected",
        [
            ("auto", True, "gpu"),
            ("auto", False, "fast"),
            ("current", True, "current"),
            ("gpu", False, "fast"),  # フォールバック
            ("gpu", True, "gpu"),
            ("fast", True, "fast"),
        ],
    )
    def test_resolve_pipeline(self, service, requested, use_gpu, expected):
        assert service.resolve_pipeline(requested, use_gpu) == expected


class TestIInferenceServiceResolvePaths:
    """resolve_paths のテスト."""

    def test_resolve_paths_prioritize_request(self, service, tmp_path):
        data_path = tmp_path / "data"
        data_path.mkdir()
        output_dir = tmp_path / "out"
        model_path = tmp_path / "model.pth"

        request = InferenceCliRequest(
            model_path=model_path,
            data_path=data_path,
            output_dir=output_dir,
            requested_pipeline="auto",
        )
        resolved = service.resolve_paths(request, config={})

        assert resolved.data_path == data_path
        assert resolved.output_dir == output_dir
        assert output_dir.exists()

    def test_resolve_paths_from_config(self, service, tmp_path):
        data_path = tmp_path / "data_from_config"
        data_path.mkdir()
        model_path = tmp_path / "model.pth"

        request = InferenceCliRequest(
            model_path=model_path,
            data_path=None,
            output_dir=None,
            requested_pipeline="auto",
        )
        config = {"val_data_root": str(data_path)}

        # we mock get_default_output_base_dir to avoid complex side effects
        with pytest.MonkeyPatch().context() as m:
            m.setattr(
                "pochitrain.inference.services.interfaces.get_default_output_base_dir",
                lambda _: tmp_path / "work_dirs",
            )
            resolved = service.resolve_paths(request, config)

        assert resolved.data_path == data_path
        assert "work_dirs" in str(resolved.output_dir)

    def test_resolve_paths_raises_error_if_no_data(self, service, tmp_path):
        request = InferenceCliRequest(
            model_path=tmp_path / "model.pth",
            data_path=None,
            output_dir=None,
            requested_pipeline="auto",
        )
        with pytest.raises(ValueError, match="--data を指定するか"):
            service.resolve_paths(request, config={})


class TestIInferenceServiceResolveRuntimeOptions:
    """resolve_runtime_options のテスト."""

    @pytest.mark.parametrize(
        "config, pipeline, use_gpu, expected_batch, expected_workers, expected_pin",
        [
            ({}, "gpu", True, 1, 0, True),
            (
                {"batch_size": 16, "num_workers": 4, "infer_pin_memory": False},
                "fast",
                False,
                16,
                4,
                False,
            ),
        ],
    )
    def test_resolve_runtime_options(
        self,
        service,
        config,
        pipeline,
        use_gpu,
        expected_batch,
        expected_workers,
        expected_pin,
    ):
        options = service.resolve_runtime_options(config, pipeline, use_gpu)
        assert options.batch_size == expected_batch
        assert options.num_workers == expected_workers
        assert options.pin_memory == expected_pin
        assert options.use_gpu == use_gpu
        assert options.use_gpu_pipeline == (pipeline == "gpu")


class TestIInferenceServiceBuildRequest:
    """build_runtime_execution_request のテスト."""

    def test_build_runtime_execution_request_basic(self, service):
        from unittest.mock import MagicMock

        import torch

        adapter = MagicMock()
        adapter.device = "cpu"

        request = service.build_runtime_execution_request(
            data_loader=MagicMock(),
            runtime_adapter=adapter,
            use_gpu_pipeline=False,
        )
        assert request.execution_request.use_gpu_pipeline is False

    def test_build_runtime_execution_request_with_gpu_pipeline(self, service):
        from unittest.mock import MagicMock

        import torch

        adapter = MagicMock()
        adapter.device = "cpu"

        request = service.build_runtime_execution_request(
            data_loader=MagicMock(),
            runtime_adapter=adapter,
            use_gpu_pipeline=True,
            norm_mean=[0.5, 0.5, 0.5],
            norm_std=[0.5, 0.5, 0.5],
        )
        assert request.execution_request.use_gpu_pipeline is True
        assert request.execution_request.mean_255 is not None

    def test_build_runtime_execution_request_raises_if_missing_norm(self, service):
        from unittest.mock import MagicMock

        with pytest.raises(ValueError, match="normalize パラメータが必要"):
            service.build_runtime_execution_request(
                data_loader=MagicMock(),
                runtime_adapter=MagicMock(),
                use_gpu_pipeline=True,
            )


class TestIInferenceServiceRun:
    """run のテスト."""

    def test_run_delegates_to_execution_service(self, service):
        from unittest.mock import MagicMock

        from pochitrain.inference.types.execution_types import (
            ExecutionRequest,
            ExecutionResult,
        )
        from pochitrain.inference.types.orchestration_types import (
            RuntimeExecutionRequest,
        )

        exec_service = MagicMock()
        exec_service.run.return_value = ExecutionResult(
            predictions=[1, 0],
            confidences=[0.9, 0.8],
            true_labels=[1, 1],
            total_inference_time_ms=10.0,
            total_samples=2,
            warmup_samples=0,
            e2e_total_time_ms=20.0,
        )

        request = RuntimeExecutionRequest(
            data_loader=MagicMock(),
            runtime_adapter=MagicMock(),
            execution_request=ExecutionRequest(use_gpu_pipeline=False),
        )

        result = service.run(request, execution_service=exec_service)

        assert result.correct == 1
        assert result.num_samples == 2
        assert result.avg_time_per_image == 5.0


class TestIInferenceServiceAggregate:
    """aggregate_and_export のテスト."""

    def test_aggregate_and_export_calls_subservices(self, service, tmp_path):
        from unittest.mock import MagicMock, patch

        from pochitrain.inference.types.orchestration_types import InferenceRunResult

        mock_dataset = MagicMock()
        mock_dataset.get_file_paths.return_value = ["img.jpg"]
        mock_dataset.get_classes.return_value = ["cat"]

        run_result = InferenceRunResult(
            predictions=[0],
            confidences=[0.9],
            true_labels=[0],
            num_samples=1,
            correct=1,
            avg_time_per_image=10.0,
            total_samples=1,
            warmup_samples=0,
            avg_total_time_per_image=15.0,
        )

        with (
            patch(
                "pochitrain.inference.services.interfaces.ResultExportService"
            ) as mock_export_cls,
            patch(
                "pochitrain.inference.services.interfaces.log_inference_result"
            ) as mock_log,
        ):
            service.aggregate_and_export(
                workspace_dir=tmp_path,
                model_path=Path("model.pth"),
                data_path=Path("data"),
                dataset=mock_dataset,
                run_result=run_result,
                input_size=(3, 224, 224),
                model_info=None,
                cm_config=None,
            )

            mock_log.assert_called_once()
            mock_export_cls.return_value.export.assert_called_once()
