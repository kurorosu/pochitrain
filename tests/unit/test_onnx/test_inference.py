"""OnnxInferenceクラスのテスト."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from pochitrain.onnx.inference import OnnxInference, check_gpu_availability


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


def create_test_onnx_model(tmp_path: Path, num_classes: int = 10) -> Path:
    """テスト用ONNXモデルを作成."""
    model = SimpleModel(num_classes=num_classes)
    model.eval()

    output_path = tmp_path / "test_model.onnx"
    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    return output_path


class TestCheckGpuAvailability:
    """check_gpu_availability関数のテスト."""

    def test_check_gpu_availability_returns_bool(self):
        """戻り値がboolであることを確認."""
        result = check_gpu_availability()
        assert isinstance(result, bool)

    def test_check_gpu_availability_with_cuda(self):
        """CUDAプロバイダーが利用可能な場合のテスト（モック）."""
        with patch("onnxruntime.get_available_providers") as mock_get:
            mock_get.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            assert check_gpu_availability() is True

    def test_check_gpu_availability_without_cuda(self):
        """CUDAプロバイダーが利用不可の場合のテスト（モック）."""
        with patch("onnxruntime.get_available_providers") as mock_get:
            mock_get.return_value = ["CPUExecutionProvider"]
            assert check_gpu_availability() is False


class TestOnnxInferenceInit:
    """OnnxInference初期化のテスト."""

    def test_init_cpu(self, tmp_path):
        """CPU初期化のテスト."""
        model_path = create_test_onnx_model(tmp_path)
        inference = OnnxInference(model_path, use_gpu=False)

        assert inference.model_path == model_path
        assert inference.use_gpu is False
        assert inference.session is not None

    def test_init_gpu_not_available(self, tmp_path):
        """GPU指定時にCUDAが利用不可の場合のテスト."""
        model_path = create_test_onnx_model(tmp_path)

        with patch(
            "pochitrain.onnx.inference.check_gpu_availability", return_value=False
        ):
            inference = OnnxInference(model_path, use_gpu=True)
            # GPUが利用不可のためCPUにフォールバック
            assert inference.use_gpu is False


class TestOnnxInferenceRun:
    """OnnxInference.run()のテスト."""

    def test_run_single_image(self, tmp_path):
        """単一画像での推論テスト."""
        model_path = create_test_onnx_model(tmp_path, num_classes=5)
        inference = OnnxInference(model_path, use_gpu=False)

        # テスト入力
        images = np.random.randn(1, 3, 224, 224).astype(np.float32)
        predicted, confidence = inference.run(images)

        assert predicted.shape == (1,)
        assert confidence.shape == (1,)
        assert 0 <= predicted[0] < 5
        assert 0 <= confidence[0] <= 1

    def test_run_batch_images(self, tmp_path):
        """バッチ画像での推論テスト."""
        model_path = create_test_onnx_model(tmp_path, num_classes=5)
        inference = OnnxInference(model_path, use_gpu=False)

        # テスト入力（バッチサイズ4）
        images = np.random.randn(4, 3, 224, 224).astype(np.float32)
        predicted, confidence = inference.run(images)

        assert predicted.shape == (4,)
        assert confidence.shape == (4,)
        for i in range(4):
            assert 0 <= predicted[i] < 5
            assert 0 <= confidence[i] <= 1

    def test_run_confidence_sum_to_one(self, tmp_path):
        """信頼度がsoftmax適用後であることを確認."""
        model_path = create_test_onnx_model(tmp_path, num_classes=5)
        inference = OnnxInference(model_path, use_gpu=False)

        images = np.random.randn(1, 3, 224, 224).astype(np.float32)
        predicted, confidence = inference.run(images)

        # 信頼度は最大確率なので0-1の範囲
        assert 0.2 <= confidence[0] <= 1.0  # 5クラスの場合、最低でも1/5=0.2


class TestOnnxInferenceGetProviders:
    """OnnxInference.get_providers()のテスト."""

    def test_get_providers_returns_list(self, tmp_path):
        """プロバイダーリストを返すことを確認."""
        model_path = create_test_onnx_model(tmp_path)
        inference = OnnxInference(model_path, use_gpu=False)

        providers = inference.get_providers()

        assert isinstance(providers, list)
        assert len(providers) > 0
        assert "CPUExecutionProvider" in providers
