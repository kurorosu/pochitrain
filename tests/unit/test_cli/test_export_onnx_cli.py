"""`export-onnx` CLIのテスト.

失敗系は最小限にし、成功系で委譲内容と引数伝播を検証する.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import pochitrain.cli.export_onnx as export_onnx_cli

RunExport = Callable[..., Any]


class FakeOnnxExporter:
    """`OnnxExporter` の呼び出しを記録するダミー."""

    instances: list["FakeOnnxExporter"] = []

    def __init__(self, device: Any) -> None:
        self.device = device
        self.load_model_args: tuple[Path, str, int] | None = None
        self.export_args: dict[str, Any] | None = None
        self.verify_args: dict[str, Any] | None = None
        FakeOnnxExporter.instances.append(self)

    def load_model(self, model_path: Path, model_name: str, num_classes: int) -> None:
        self.load_model_args = (model_path, model_name, num_classes)

    def export(
        self,
        output_path: Path,
        input_size: tuple[int, int],
        opset_version: int,
    ) -> None:
        self.export_args = {
            "output_path": output_path,
            "input_size": input_size,
            "opset_version": opset_version,
        }

    def verify(self, onnx_path: Path, input_size: tuple[int, int]) -> bool:
        self.verify_args = {"onnx_path": onnx_path, "input_size": input_size}
        return True


class FakeOnnxExporterVerifyFail(FakeOnnxExporter):
    """verify が失敗するダミー."""

    def verify(self, onnx_path: Path, input_size: tuple[int, int]) -> bool:
        self.verify_args = {"onnx_path": onnx_path, "input_size": input_size}
        return False


@pytest.fixture
def run_export(monkeypatch: pytest.MonkeyPatch) -> RunExport:
    """CLI実行ヘルパー."""

    def _run(
        argv_tail: list[str],
        exporter_cls: type[Any] = FakeOnnxExporter,
    ) -> Any:
        FakeOnnxExporter.instances.clear()
        monkeypatch.setattr(export_onnx_cli, "OnnxExporter", exporter_cls)
        monkeypatch.setattr("sys.argv", ["export-onnx", *argv_tail])
        export_onnx_cli.main()
        return exporter_cls.instances[0]

    return _run


class TestExportOnnxCli:
    """`export-onnx` のCLI検証."""

    def test_model_path_must_exist(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """存在しないモデルパスなら終了する."""
        missing = tmp_path / "missing.pth"
        monkeypatch.setattr(
            "sys.argv",
            [
                "export-onnx",
                str(missing),
                "--num-classes",
                "2",
                "--input-size",
                "224",
                "224",
            ],
        )
        with pytest.raises(SystemExit):
            export_onnx_cli.main()

    def test_num_classes_is_required_without_config(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """config未使用時は `--num-classes` が必須である."""
        model_path = tmp_path / "model.pth"
        model_path.write_text("dummy", encoding="utf-8")

        monkeypatch.setattr(
            "sys.argv",
            [
                "export-onnx",
                str(model_path),
                "--input-size",
                "224",
                "224",
            ],
        )
        with pytest.raises(SystemExit):
            export_onnx_cli.main()

    def test_default_values_are_applied(
        self,
        tmp_path: Path,
        run_export: RunExport,
    ) -> None:
        """デフォルト値と `--skip-verify` が反映される."""
        model_path = tmp_path / "model.pth"
        model_path.write_text("dummy", encoding="utf-8")

        instance = run_export(
            [
                str(model_path),
                "--num-classes",
                "2",
                "--input-size",
                "224",
                "224",
                "--skip-verify",
            ]
        )

        assert instance.load_model_args is not None
        assert instance.load_model_args[1] == "resnet18"
        assert instance.load_model_args[2] == 2
        assert instance.export_args is not None
        assert instance.export_args["input_size"] == (224, 224)
        assert instance.export_args["opset_version"] == 17
        assert instance.verify_args is None

    def test_verify_runs_when_not_skipped(
        self,
        tmp_path: Path,
        run_export: RunExport,
    ) -> None:
        """`--skip-verify` なしでは verify が実行される."""
        model_path = tmp_path / "model.pth"
        model_path.write_text("dummy", encoding="utf-8")

        instance = run_export(
            [
                str(model_path),
                "--num-classes",
                "2",
                "--input-size",
                "320",
                "640",
            ]
        )

        assert instance.verify_args is not None
        assert instance.verify_args["input_size"] == (320, 640)

    def test_uses_model_settings_from_explicit_config(
        self,
        tmp_path: Path,
        run_export: RunExport,
    ) -> None:
        """`--config` の model_name/num_classes が load_model に伝播する."""
        model_path = tmp_path / "model.pth"
        model_path.write_text("dummy", encoding="utf-8")
        config_path = tmp_path / "config.py"
        config_path.write_text(
            'model_name = "resnet50"\nnum_classes = 7\n',
            encoding="utf-8",
        )

        instance = run_export(
            [
                str(model_path),
                "--config",
                str(config_path),
                "--input-size",
                "224",
                "224",
                "--skip-verify",
            ]
        )

        assert instance.load_model_args is not None
        assert instance.load_model_args[1] == "resnet50"
        assert instance.load_model_args[2] == 7

    def test_auto_discovers_config_next_to_workdir_model(
        self,
        tmp_path: Path,
        run_export: RunExport,
    ) -> None:
        """`work_dir/models/*.pth` から `work_dir/config.py` を自動探索できる."""
        work_dir = tmp_path / "work_dirs" / "20260223_001"
        models_dir = work_dir / "models"
        models_dir.mkdir(parents=True)
        model_path = models_dir / "best_epoch40.pth"
        model_path.write_text("dummy", encoding="utf-8")
        config_path = work_dir / "config.py"
        config_path.write_text(
            'model_name = "resnet34"\nnum_classes = 5\n',
            encoding="utf-8",
        )

        instance = run_export(
            [
                str(model_path),
                "--input-size",
                "224",
                "224",
                "--skip-verify",
            ]
        )

        assert instance.load_model_args is not None
        assert instance.load_model_args[1] == "resnet34"
        assert instance.load_model_args[2] == 5

    def test_cli_num_classes_overrides_config(
        self,
        tmp_path: Path,
        run_export: RunExport,
    ) -> None:
        """`--num-classes` 指定時は config の num_classes より優先される."""
        model_path = tmp_path / "model.pth"
        model_path.write_text("dummy", encoding="utf-8")
        config_path = tmp_path / "config.py"
        config_path.write_text(
            'model_name = "resnet50"\nnum_classes = 7\n',
            encoding="utf-8",
        )

        instance = run_export(
            [
                str(model_path),
                "--config",
                str(config_path),
                "--num-classes",
                "3",
                "--input-size",
                "224",
                "224",
                "--skip-verify",
            ]
        )

        assert instance.load_model_args is not None
        assert instance.load_model_args[1] == "resnet50"
        assert instance.load_model_args[2] == 3

    def test_verify_failure_exits_with_error(
        self,
        tmp_path: Path,
        run_export: RunExport,
    ) -> None:
        """verify が False の場合は終了する."""
        model_path = tmp_path / "model.pth"
        model_path.write_text("dummy", encoding="utf-8")

        with pytest.raises(SystemExit):
            run_export(
                [
                    str(model_path),
                    "--num-classes",
                    "2",
                    "--input-size",
                    "224",
                    "224",
                ],
                exporter_cls=FakeOnnxExporterVerifyFail,
            )
