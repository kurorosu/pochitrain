"""`export-onnx` CLIのテスト.

実モジュールの `main` を通し, 引数チェックと呼び出し連携を検証する.
"""

from pathlib import Path

import pytest

import pochitrain.cli.export_onnx as export_onnx_cli


class FakeOnnxExporter:
    """`OnnxExporter` を置き換えるテストダブル."""

    instances: list["FakeOnnxExporter"] = []

    def __init__(self, device) -> None:
        """呼び出し履歴を保持する初期化."""
        self.device = device
        self.load_model_args: tuple | None = None
        self.export_args: dict | None = None
        self.verify_args: dict | None = None
        FakeOnnxExporter.instances.append(self)

    def load_model(self, model_path: Path, model_name: str, num_classes: int) -> None:
        """モデル読み込み引数を記録."""
        self.load_model_args = (model_path, model_name, num_classes)

    def export(
        self,
        output_path: Path,
        input_size: tuple[int, int],
        opset_version: int,
    ) -> None:
        """ONNX出力引数を記録."""
        self.export_args = {
            "output_path": output_path,
            "input_size": input_size,
            "opset_version": opset_version,
        }

    def verify(self, onnx_path: Path, input_size: tuple[int, int]) -> bool:
        """検証呼び出しを記録して成功を返す."""
        self.verify_args = {"onnx_path": onnx_path, "input_size": input_size}
        return True


class TestExportOnnxCli:
    """`export-onnx` のCLI挙動検証."""

    def setup_method(self) -> None:
        """テスト間でダミーインスタンスを初期化."""
        FakeOnnxExporter.instances.clear()

    def test_input_size_required(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`--input-size` 未指定時にエラー終了することを確認."""
        monkeypatch.setattr(
            "sys.argv", ["export-onnx", "model.pth", "--num-classes", "2"]
        )
        with pytest.raises(SystemExit):
            export_onnx_cli.main()

    def test_model_path_must_exist(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """存在しないモデルパスが拒否されることを確認."""
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
        """configから取得できない場合は `--num-classes` が必須であることを確認."""
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
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """既定値と `--skip-verify` 分岐を確認."""
        model_path = tmp_path / "model.pth"
        model_path.write_text("dummy", encoding="utf-8")

        monkeypatch.setattr(export_onnx_cli, "OnnxExporter", FakeOnnxExporter)
        monkeypatch.setattr(
            "sys.argv",
            [
                "export-onnx",
                str(model_path),
                "--num-classes",
                "2",
                "--input-size",
                "224",
                "224",
                "--skip-verify",
            ],
        )

        export_onnx_cli.main()

        assert len(FakeOnnxExporter.instances) == 1
        instance = FakeOnnxExporter.instances[0]
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
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`--skip-verify` なしでは verify が呼ばれることを確認."""
        model_path = tmp_path / "model.pth"
        model_path.write_text("dummy", encoding="utf-8")

        monkeypatch.setattr(export_onnx_cli, "OnnxExporter", FakeOnnxExporter)
        monkeypatch.setattr(
            "sys.argv",
            [
                "export-onnx",
                str(model_path),
                "--num-classes",
                "2",
                "--input-size",
                "320",
                "640",
            ],
        )

        export_onnx_cli.main()

        instance = FakeOnnxExporter.instances[0]
        assert instance.verify_args is not None
        assert instance.verify_args["input_size"] == (320, 640)
