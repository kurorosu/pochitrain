"""OnnxInference クラスのユニットテスト."""

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

pytest.importorskip("onnx")
onnxruntime = pytest.importorskip("onnxruntime")

from pochitrain.onnx.inference import OnnxInference, check_gpu_availability


class SimpleModel(nn.Module):
    """テスト用のシンプルなモデル."""

    def __init__(self, num_classes: int = 10) -> None:
        """モデルを初期化する.

        Args:
            num_classes: 出力クラス数.
        """
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播を行う.

        Args:
            x: 入力テンソル.

        Returns:
            ログイット.
        """
        x = self.conv(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def create_test_onnx_model(tmp_path: Path, num_classes: int = 10) -> Path:
    """テスト用 ONNX モデルを作成する.

    Args:
        tmp_path: 一時ディレクトリ.
        num_classes: 出力クラス数.

    Returns:
        生成した ONNX ファイルパス.
    """
    model = SimpleModel(num_classes=num_classes)
    model.eval()

    output_path = tmp_path / "test_model.onnx"
    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        (dummy_input,),
        str(output_path),
        export_params=True,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    return output_path


class TestCheckGpuAvailability:
    """check_gpu_availability 関数のテスト."""

    def test_check_gpu_availability_returns_bool(self) -> None:
        """戻り値が bool であることを確認する."""
        result = check_gpu_availability()
        assert isinstance(result, bool)

    def test_check_gpu_availability_with_cuda(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CUDA プロバイダーがある場合に True を返すことを確認する."""
        monkeypatch.setattr(
            onnxruntime,
            "get_available_providers",
            lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        assert check_gpu_availability() is True

    def test_check_gpu_availability_without_cuda(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CUDA プロバイダーがない場合に False を返すことを確認する."""
        monkeypatch.setattr(
            onnxruntime,
            "get_available_providers",
            lambda: ["CPUExecutionProvider"],
        )
        assert check_gpu_availability() is False


class TestOnnxInferenceInit:
    """OnnxInference 初期化のテスト."""

    def test_init_cpu(self, tmp_path: Path) -> None:
        """CPU 初期化が成功することを確認する."""
        model_path = create_test_onnx_model(tmp_path)
        inference = OnnxInference(model_path, use_gpu=False)

        assert inference.model_path == model_path
        assert inference.use_gpu is False
        assert inference.session is not None

    def test_init_gpu_not_available(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """GPU 指定時に CUDA 未対応なら CPU へフォールバックすることを確認する."""
        model_path = create_test_onnx_model(tmp_path)

        monkeypatch.setattr(
            "pochitrain.onnx.inference.check_gpu_availability",
            lambda: False,
        )

        inference = OnnxInference(model_path, use_gpu=True)
        assert inference.use_gpu is False


class TestOnnxInferenceRun:
    """OnnxInference.run のテスト."""

    def test_run_single_image(self, tmp_path: Path) -> None:
        """単一画像の推論結果形状と値域を確認する."""
        model_path = create_test_onnx_model(tmp_path, num_classes=5)
        inference = OnnxInference(model_path, use_gpu=False)

        images = np.random.randn(1, 3, 224, 224).astype(np.float32)
        predicted, confidence = inference.run(images)

        assert predicted.shape == (1,)
        assert confidence.shape == (1,)
        assert 0 <= predicted[0] < 5
        assert 0 <= confidence[0] <= 1

    def test_run_batch_images(self, tmp_path: Path) -> None:
        """バッチ画像推論の結果形状と値域を確認する."""
        model_path = create_test_onnx_model(tmp_path, num_classes=5)
        inference = OnnxInference(model_path, use_gpu=False)

        images = np.random.randn(4, 3, 224, 224).astype(np.float32)
        predicted, confidence = inference.run(images)

        assert predicted.shape == (4,)
        assert confidence.shape == (4,)
        for i in range(4):
            assert 0 <= predicted[i] < 5
            assert 0 <= confidence[i] <= 1

    def test_run_confidence_range(self, tmp_path: Path) -> None:
        """最大信頼度が想定範囲に収まることを確認する."""
        model_path = create_test_onnx_model(tmp_path, num_classes=5)
        inference = OnnxInference(model_path, use_gpu=False)

        images = np.random.randn(1, 3, 224, 224).astype(np.float32)
        _, confidence = inference.run(images)

        assert 0.2 <= confidence[0] <= 1.0


class TestOnnxInferenceGetProviders:
    """OnnxInference.get_providers のテスト."""

    def test_get_providers_returns_list(self, tmp_path: Path) -> None:
        """利用プロバイダー一覧を取得できることを確認する."""
        model_path = create_test_onnx_model(tmp_path)
        inference = OnnxInference(model_path, use_gpu=False)

        providers = inference.get_providers()

        assert isinstance(providers, list)
        assert len(providers) > 0
        assert "CPUExecutionProvider" in providers
