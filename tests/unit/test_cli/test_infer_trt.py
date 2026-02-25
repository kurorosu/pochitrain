"""infer_trt CLIの導線テスト."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from pochitrain.cli.infer_trt import main
from pochitrain.inference.types.execution_types import ExecutionResult


@pytest.fixture
def infer_trt_env(tmp_path):
    """TRT推論用のテスト環境を作成する."""
    work_dir = tmp_path / "work_dirs" / "20260225_001"
    models_dir = work_dir / "models"
    models_dir.mkdir(parents=True)

    engine_path = models_dir / "model.engine"
    engine_path.touch()

    data_dir = tmp_path / "data" / "class_a"
    data_dir.mkdir(parents=True)
    img = Image.new("RGB", (32, 32), color="red")
    img.save(str(data_dir / "dummy.jpg"))

    config_path = work_dir / "config.py"
    config_path.write_text(
        f'val_data_root = r"{tmp_path / "data"}"\n' 'model_name = "resnet18"\n',
        encoding="utf-8",
    )

    return engine_path, tmp_path / "data", config_path


def test_main_delegates_to_service_and_writes_outputs(infer_trt_env, tmp_path):
    """main() が Service を通じて推論を実行し、結果を出力することを確認する (古典派スタイル)."""
    engine_path, data_path, _ = infer_trt_env
    output_dir = tmp_path / "output"

    # Mock TensorRTInference to avoid real engine loading
    mock_inference = MagicMock()
    mock_inference.input_shape = (1, 3, 32, 32)
    mock_inference.get_input_shape.return_value = (1, 3, 32, 32)

    # Mock ExecutionService.run to return a dummy result
    dummy_exec_result = ExecutionResult(
        predictions=[0],
        confidences=[0.9],
        true_labels=[0],
        total_inference_time_ms=10.0,
        total_samples=1,
        warmup_samples=0,
        e2e_total_time_ms=20.0,
    )

    with (
        patch(
            "sys.argv",
            [
                "infer-trt",
                str(engine_path),
                "--data",
                str(data_path),
                "-o",
                str(output_dir),
                "--benchmark-json",
            ],
        ),
        patch("pochitrain.tensorrt.TensorRTInference", return_value=mock_inference),
        patch(
            "pochitrain.inference.services.execution_service.ExecutionService.run",
            return_value=dummy_exec_result,
        ),
    ):
        main()

    # Verify outputs
    assert (output_dir / "tensorrt_inference_summary.txt").exists()
    assert (output_dir / "tensorrt_inference_results.csv").exists()
    assert (output_dir / "benchmark_result.json").exists()

    with (output_dir / "benchmark_result.json").open("r", encoding="utf-8") as f:
        payload = json.load(f)
        assert payload["runtime"] == "tensorrt"
        assert isinstance(payload["env_name"], str)
        assert payload["env_name"] != ""


def test_main_exits_on_nonexistent_engine(tmp_path):
    """存在しないエンジンファイル指定で SystemExit することを確認する."""
    with patch("sys.argv", ["infer-trt", str(tmp_path / "none.engine")]):
        with pytest.raises(SystemExit):
            main()
