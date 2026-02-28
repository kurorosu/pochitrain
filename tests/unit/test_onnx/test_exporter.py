"""OnnxExporter クラスのユニットテスト."""

from pathlib import Path

import pytest
import torch
import torch.nn as nn

pytest.importorskip("onnx")
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:You are using the legacy TorchScript-based ONNX export.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:The feature will be removed\\. Please remove usage of this function:DeprecationWarning"
    ),
]

from pochitrain.onnx.exporter import OnnxExporter

from .conftest import SimpleModel


class TestOnnxExporterInit:
    """OnnxExporter 初期化のテスト."""

    def test_init_default(self) -> None:
        """デフォルト初期化の値を確認する."""
        exporter = OnnxExporter()
        assert exporter.model is None
        assert exporter.device == torch.device("cpu")

    def test_init_with_model(self) -> None:
        """モデル指定初期化を確認する."""
        model = SimpleModel()
        exporter = OnnxExporter(model=model)
        assert exporter.model is model

    def test_init_with_device(self) -> None:
        """デバイス指定初期化を確認する."""
        device = torch.device("cpu")
        exporter = OnnxExporter(device=device)
        assert exporter.device == device


class TestOnnxExporterExport:
    """OnnxExporter.export のテスト."""

    def test_export_success(self, tmp_path: Path) -> None:
        """正常なエクスポートを確認する."""
        model = SimpleModel()
        exporter = OnnxExporter(model=model)

        output_path = tmp_path / "model.onnx"
        result = exporter.export(output_path, input_size=(224, 224))

        assert result == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_export_without_model_raises_error(self, tmp_path: Path) -> None:
        """モデル未設定で ValueError になることを確認する."""
        exporter = OnnxExporter()

        with pytest.raises(ValueError, match="モデル"):
            exporter.export(tmp_path / "model.onnx", input_size=(224, 224))

    def test_export_custom_opset(self, tmp_path: Path) -> None:
        """カスタム opset_version を指定できることを確認する."""
        model = SimpleModel()
        exporter = OnnxExporter(model=model)

        output_path = tmp_path / "model.onnx"
        result = exporter.export(output_path, input_size=(224, 224), opset_version=14)

        assert result == output_path
        assert output_path.exists()

    def test_export_custom_input_size(self, tmp_path: Path) -> None:
        """カスタム入力サイズ指定を確認する."""
        model = SimpleModel()
        exporter = OnnxExporter(model=model)

        output_path = tmp_path / "model.onnx"
        result = exporter.export(output_path, input_size=(128, 128))

        assert result == output_path
        assert output_path.exists()


class TestOnnxExporterVerify:
    """OnnxExporter.verify のテスト."""

    def test_verify_success(self, tmp_path: Path) -> None:
        """正常な検証が True を返すことを確認する."""
        model = SimpleModel()
        exporter = OnnxExporter(model=model)

        output_path = tmp_path / "model.onnx"
        exporter.export(output_path, input_size=(224, 224))

        result = exporter.verify(output_path, input_size=(224, 224))
        assert result is True

    def test_verify_without_model_raises_error(self, tmp_path: Path) -> None:
        """モデル未設定で verify が ValueError になることを確認する."""
        exporter = OnnxExporter()

        with pytest.raises(ValueError, match="モデル"):
            exporter.verify(tmp_path / "model.onnx", input_size=(224, 224))


class TestOnnxExporterLoadModel:
    """OnnxExporter.load_model のテスト."""

    def test_load_model_with_state_dict(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """model_state_dict 形式チェックポイントを読み込めることを確認する."""
        call_args: dict[str, object] = {}

        def _fake_create_model(
            model_name: str, num_classes: int, pretrained: bool = False
        ) -> nn.Module:
            call_args["model_name"] = model_name
            call_args["num_classes"] = num_classes
            call_args["pretrained"] = pretrained
            return SimpleModel(num_classes=num_classes)

        monkeypatch.setattr("pochitrain.onnx.exporter.create_model", _fake_create_model)

        checkpoint_model = SimpleModel(num_classes=5)
        checkpoint = {
            "model_state_dict": checkpoint_model.state_dict(),
            "epoch": 10,
            "best_accuracy": 95.0,
        }
        checkpoint_path = tmp_path / "checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)

        exporter = OnnxExporter()
        exporter.load_model(checkpoint_path, model_name="resnet18", num_classes=5)

        assert exporter.model is not None
        assert call_args == {
            "model_name": "resnet18",
            "num_classes": 5,
            "pretrained": False,
        }

    def test_load_model_raw_state_dict(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """生の state_dict 形式チェックポイントを読み込めることを確認する."""
        call_args: dict[str, object] = {}

        def _fake_create_model(
            model_name: str, num_classes: int, pretrained: bool = False
        ) -> nn.Module:
            call_args["model_name"] = model_name
            call_args["num_classes"] = num_classes
            call_args["pretrained"] = pretrained
            return SimpleModel(num_classes=num_classes)

        monkeypatch.setattr("pochitrain.onnx.exporter.create_model", _fake_create_model)

        checkpoint_model = SimpleModel(num_classes=5)
        checkpoint_path = tmp_path / "checkpoint.pth"
        torch.save(checkpoint_model.state_dict(), checkpoint_path)

        exporter = OnnxExporter()
        exporter.load_model(checkpoint_path, model_name="resnet18", num_classes=5)

        assert exporter.model is not None
        assert call_args == {
            "model_name": "resnet18",
            "num_classes": 5,
            "pretrained": False,
        }
