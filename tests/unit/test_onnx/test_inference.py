"""OnnxInference クラスのユニットテスト."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

pytest.importorskip("onnx")
onnxruntime = pytest.importorskip("onnxruntime")
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:You are using the legacy TorchScript-based ONNX export.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:The feature will be removed\\. Please remove usage of this function:DeprecationWarning"
    ),
]

from pochitrain.onnx.inference import OnnxInference, check_gpu_availability

from .conftest import SimpleModel


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
        dynamo=False,
        export_params=True,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    return output_path


def create_cpu_inference(tmp_path: Path, num_classes: int = 10) -> OnnxInference:
    """CPU モードの OnnxInference を生成する."""
    model_path = create_test_onnx_model(tmp_path, num_classes=num_classes)
    return OnnxInference(model_path, use_gpu=False)


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

    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_run_images_shape_and_range(self, tmp_path: Path, batch_size: int) -> None:
        """単一/複数バッチで推論結果の形状と値域を確認する."""
        inference = create_cpu_inference(tmp_path, num_classes=5)

        images = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
        predicted, confidence = inference.run(images)

        assert predicted.shape == (batch_size,)
        assert confidence.shape == (batch_size,)
        for i in range(batch_size):
            assert 0 <= predicted[i] < 5
            assert 0 <= confidence[i] <= 1

    def test_run_confidence_range(self, tmp_path: Path) -> None:
        """最大信頼度が想定範囲に収まることを確認する."""
        inference = create_cpu_inference(tmp_path, num_classes=5)

        images = np.random.randn(1, 3, 224, 224).astype(np.float32)
        _, confidence = inference.run(images)

        assert 0.2 <= confidence[0] <= 1.0


class TestOnnxInferenceGetProviders:
    """OnnxInference.get_providers のテスト."""

    def test_get_providers_returns_list(self, tmp_path: Path) -> None:
        """利用プロバイダー一覧を取得できることを確認する."""
        inference = create_cpu_inference(tmp_path)

        providers = inference.get_providers()

        assert isinstance(providers, list)
        assert len(providers) > 0
        assert "CPUExecutionProvider" in providers


class _FakeIOBinding:
    """IO Binding の最小スタブ."""

    def __init__(self) -> None:
        self.bound_buffer_ptr: int | None = None
        self.bound_shape: tuple[int, ...] | None = None
        self.bound_output_name: str | None = None

    def bind_cpu_input(self, name: str, images: np.ndarray) -> None:
        _ = (name, images)

    def bind_input(
        self,
        name: str,
        device_type: str,
        device_id: int,
        element_type: type[np.float32],
        shape: tuple[int, ...],
        buffer_ptr: int,
    ) -> None:
        _ = (name, device_type, device_id, element_type)
        self.bound_shape = shape
        self.bound_buffer_ptr = buffer_ptr

    def bind_output(self, name: str, device_type: str) -> None:
        _ = device_type
        self.bound_output_name = name

    def copy_outputs_to_cpu(self) -> list[np.ndarray]:
        return [np.array([[1.0, 0.0]], dtype=np.float32)]


class _FakeSession:
    """ONNX Runtime セッションの最小スタブ."""

    def __init__(self, io_binding: _FakeIOBinding) -> None:
        self._io_binding = io_binding
        self.run_called = False

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="input")]

    def get_outputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="output")]

    def io_binding(self) -> _FakeIOBinding:
        return self._io_binding

    def run_with_iobinding(self, io_binding: _FakeIOBinding) -> None:
        assert io_binding is self._io_binding
        self.run_called = True


class TestOnnxInferenceGpuInputBinding:
    """GPU入力バインディングの寿命管理テスト."""

    def test_set_input_gpu_keeps_tensor_until_output(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """出力取得まで入力テンソル参照を保持し, 取得後に解放する."""
        fake_binding = _FakeIOBinding()
        fake_session = _FakeSession(fake_binding)
        monkeypatch.setattr(
            OnnxInference,
            "_create_session",
            lambda self: fake_session,
        )

        inference = OnnxInference(Path("dummy.onnx"), use_gpu=True)
        tensor = torch.randn(1, 3, 8, 8).transpose(2, 3)
        inference.set_input_gpu(tensor)

        assert inference._bound_input_tensor is not None
        assert inference._bound_input_tensor.is_contiguous()
        assert fake_binding.bound_shape == tuple(inference._bound_input_tensor.shape)
        assert fake_binding.bound_buffer_ptr == inference._bound_input_tensor.data_ptr()

        inference.run_pure()
        assert fake_session.run_called is True
        assert inference._bound_input_tensor is not None

        _ = inference.get_output()
        assert inference._bound_input_tensor is None
