"""infer_onnx CLIの入口導線テスト.

実際のONNX推論ロジックは test_onnx/test_inference.py でテスト済み.
"""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from PIL import Image

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:You are using the legacy TorchScript-based ONNX export.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:The feature will be removed\\. Please remove usage of this function:DeprecationWarning"
    ),
]


class _SimpleModel(nn.Module):
    """GPUフォールバックテスト用の最小モデル."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 2)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        return self.fc(x.view(x.size(0), -1))


def _create_dummy_artifact(
    *,
    output_dir: Path,
    filename: str = "dummy.txt",
    **_kwargs,
) -> Path:
    """補助成果物生成を軽量化するダミー出力関数."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    output_path.write_text("dummy", encoding="utf-8")
    return output_path


@contextmanager
def _patch_fallback_runtime(argv: list[str]) -> Iterator[None]:
    """GPUフォールバック系テストの共通 patch を適用する."""
    with (
        patch("sys.argv", argv),
        patch(
            "pochitrain.onnx.inference.check_gpu_availability",
            return_value=False,
        ),
        patch(
            "pochitrain.inference.services.result_export_service.save_confusion_matrix_image",
            side_effect=_create_dummy_artifact,
        ),
        patch(
            "pochitrain.inference.services.result_export_service.save_classification_report",
            side_effect=_create_dummy_artifact,
        ),
    ):
        yield


@pytest.fixture(scope="module")
def gpu_fallback_test_env(tmp_path_factory):
    """GPUフォールバック用の共有テスト環境を作成する."""
    tmp_path = tmp_path_factory.mktemp("infer_onnx_fallback")

    work_dir = tmp_path / "work_dir"
    models_dir = work_dir / "models"
    models_dir.mkdir(parents=True)

    model = _SimpleModel()
    model.eval()
    model_path = models_dir / "model.onnx"
    torch.onnx.export(
        model,
        torch.randn(1, 3, 32, 32),
        str(model_path),
        dynamo=False,
        export_params=True,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

    for cls_name, color in [("class_a", "red"), ("class_b", "blue")]:
        cls_dir = tmp_path / "data" / cls_name
        cls_dir.mkdir(parents=True)
        img = Image.new("RGB", (32, 32), color=color)
        img.save(str(cls_dir / "dummy.jpg"))

    config_path = work_dir / "config.py"
    config_path.write_text(
        "from torchvision import transforms\n"
        f'val_data_root = r"{tmp_path / "data"}"\n'
        "val_transform = transforms.Compose([\n"
        "    transforms.Resize((32, 32)),\n"
        "    transforms.ToTensor(),\n"
        "    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),\n"
        "])\n"
        'device = "cuda"\n'
        "batch_size = 1\n"
        "num_workers = 0\n"
        "pin_memory = False\n",
        encoding="utf-8",
    )

    data_path = tmp_path / "data"
    return model_path, data_path


class TestInferOnnxMainExit:
    """main関数のSystemExitテスト."""

    def test_main_nonexistent_model_exits(self, tmp_path):
        """存在しないモデルでSystemExitが発生する."""
        from pochitrain.cli.infer_onnx import main

        fake_model = str(tmp_path / "nonexistent.onnx")
        with patch("sys.argv", ["infer-onnx", fake_model]):
            with pytest.raises(SystemExit):
                main()


class TestGpuFallbackReresolution:
    """OnnxInference の GPU フォールバック後にパイプラインが再解決されるテスト.

    OnnxInference が内部で use_gpu=False に切り替えた場合,
    CLI側で pipeline と dataset を再解決するロジックを検証する.
    main() を通して実コード経路のフォールバックを検証する.
    """

    def test_main_does_not_crash_when_cuda_ep_unavailable(
        self,
        gpu_fallback_test_env,
        tmp_path,
    ):
        """CUDA EP不可時にmain()がRuntimeErrorなく完走する."""
        from pochitrain.cli.infer_onnx import main

        model_path, data_path = gpu_fallback_test_env
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with _patch_fallback_runtime(
            [
                "infer-onnx",
                str(model_path),
                "--data",
                str(data_path),
                "-o",
                str(output_dir),
                "--pipeline",
                "gpu",
            ]
        ):
            main()

        csv_files = list(output_dir.glob("*.csv"))
        assert len(csv_files) > 0

    def test_main_fallback_uses_set_input_not_set_input_gpu(
        self,
        gpu_fallback_test_env,
        tmp_path,
    ):
        """フォールバック後にset_input_gpu()が呼ばれないことを確認."""
        from pochitrain.cli.infer_onnx import main

        model_path, data_path = gpu_fallback_test_env
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with (
            _patch_fallback_runtime(
                [
                    "infer-onnx",
                    str(model_path),
                    "--data",
                    str(data_path),
                    "-o",
                    str(output_dir),
                    "--pipeline",
                    "gpu",
                ]
            ),
            patch(
                "pochitrain.onnx.inference.OnnxInference.set_input_gpu",
                side_effect=RuntimeError("set_input_gpuが呼ばれた"),
            ) as mock_set_input_gpu,
        ):
            main()

        mock_set_input_gpu.assert_not_called()

    def test_main_writes_benchmark_json_when_enabled(
        self,
        gpu_fallback_test_env,
        tmp_path,
    ) -> None:
        """`--benchmark-json` 指定時にJSONが出力されることを確認する."""
        from pochitrain.cli.infer_onnx import main

        model_path, data_path = gpu_fallback_test_env
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with _patch_fallback_runtime(
            [
                "infer-onnx",
                str(model_path),
                "--data",
                str(data_path),
                "-o",
                str(output_dir),
                "--pipeline",
                "gpu",
                "--benchmark-json",
                "--benchmark-env-name",
                "TestEnv",
            ]
        ):
            main()

        benchmark_json_path = output_dir / "benchmark_result.json"
        assert benchmark_json_path.exists()
        payload = json.loads(benchmark_json_path.read_text(encoding="utf-8"))
        assert payload["env_name"] == "TestEnv"
        assert payload["runtime"] == "onnx"
        assert payload["pipeline"] == "fast"
