"""OnnxExporterクラスのテスト."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from pochitrain.onnx.exporter import OnnxExporter


class SimpleModel(nn.Module):
    """テスト用のシンプルなモデル."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TestOnnxExporterInit:
    """OnnxExporter初期化のテスト."""

    def test_init_default(self):
        """デフォルト初期化のテスト."""
        exporter = OnnxExporter()
        assert exporter.model is None
        assert exporter.device == torch.device("cpu")

    def test_init_with_model(self):
        """モデル指定での初期化テスト."""
        model = SimpleModel()
        exporter = OnnxExporter(model=model)
        assert exporter.model is model

    def test_init_with_device(self):
        """デバイス指定での初期化テスト."""
        device = torch.device("cpu")
        exporter = OnnxExporter(device=device)
        assert exporter.device == device


class TestOnnxExporterExport:
    """OnnxExporter.export()のテスト."""

    def test_export_success(self, tmp_path):
        """正常なエクスポートのテスト."""
        model = SimpleModel()
        exporter = OnnxExporter(model=model)

        output_path = tmp_path / "model.onnx"
        result = exporter.export(output_path, input_size=(224, 224))

        assert result == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_export_without_model_raises_error(self, tmp_path):
        """モデルなしでエクスポートするとエラーになることを確認."""
        exporter = OnnxExporter()

        with pytest.raises(ValueError, match="モデルが設定されていません"):
            exporter.export(tmp_path / "model.onnx", input_size=(224, 224))

    def test_export_custom_opset(self, tmp_path):
        """カスタムopsetバージョンでのエクスポートテスト."""
        model = SimpleModel()
        exporter = OnnxExporter(model=model)

        output_path = tmp_path / "model.onnx"
        result = exporter.export(output_path, input_size=(224, 224), opset_version=14)

        assert result == output_path
        assert output_path.exists()

    def test_export_custom_input_size(self, tmp_path):
        """カスタム入力サイズでのエクスポートテスト."""
        model = SimpleModel()
        exporter = OnnxExporter(model=model)

        output_path = tmp_path / "model.onnx"
        result = exporter.export(output_path, input_size=(128, 128))

        assert result == output_path
        assert output_path.exists()


class TestOnnxExporterVerify:
    """OnnxExporter.verify()のテスト."""

    def test_verify_success(self, tmp_path):
        """正常な検証のテスト."""
        model = SimpleModel()
        exporter = OnnxExporter(model=model)

        output_path = tmp_path / "model.onnx"
        exporter.export(output_path, input_size=(224, 224))

        result = exporter.verify(output_path, input_size=(224, 224))
        assert result is True

    def test_verify_without_model_raises_error(self, tmp_path):
        """モデルなしで検証するとエラーになることを確認."""
        exporter = OnnxExporter()

        with pytest.raises(ValueError, match="モデルが設定されていません"):
            exporter.verify(tmp_path / "model.onnx", input_size=(224, 224))


class TestOnnxExporterLoadModel:
    """OnnxExporter.load_model()のテスト."""

    def test_load_model_with_state_dict(self, tmp_path):
        """state_dict形式のチェックポイント読み込みテスト."""
        # create_modelをモックしてテスト
        with patch("pochitrain.onnx.exporter.create_model") as mock_create:
            mock_model = SimpleModel(num_classes=5)
            mock_create.return_value = mock_model

            # テスト用チェックポイントを作成
            checkpoint = {
                "model_state_dict": mock_model.state_dict(),
                "epoch": 10,
                "best_accuracy": 95.0,
            }
            checkpoint_path = tmp_path / "checkpoint.pth"
            torch.save(checkpoint, checkpoint_path)

            # 読み込みテスト
            exporter = OnnxExporter()
            exporter.load_model(checkpoint_path, model_name="resnet18", num_classes=5)

            assert exporter.model is not None
            mock_create.assert_called_once_with("resnet18", 5, pretrained=False)

    def test_load_model_raw_state_dict(self, tmp_path):
        """生のstate_dict形式のチェックポイント読み込みテスト."""
        # create_modelをモックしてテスト
        with patch("pochitrain.onnx.exporter.create_model") as mock_create:
            mock_model = SimpleModel(num_classes=5)
            mock_create.return_value = mock_model

            # テスト用チェックポイントを作成
            checkpoint_path = tmp_path / "checkpoint.pth"
            torch.save(mock_model.state_dict(), checkpoint_path)

            exporter = OnnxExporter()
            exporter.load_model(checkpoint_path, model_name="resnet18", num_classes=5)

            assert exporter.model is not None
            mock_create.assert_called_once_with("resnet18", 5, pretrained=False)
