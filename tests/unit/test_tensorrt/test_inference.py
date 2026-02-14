"""TensorRT 推論モジュールのユニットテスト.

TensorRT がない環境でも検証できるロジックと,
TensorRT の有無に応じた初期化挙動をテストする.
"""

from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest

from pochitrain.tensorrt.inference import TensorRTInference, check_tensorrt_availability


class TestCheckTensorrtAvailability:
    """check_tensorrt_availability 関数のテスト."""

    def test_returns_bool(self) -> None:
        """戻り値が bool であることを確認する."""
        result = check_tensorrt_availability()
        assert isinstance(result, bool)


class _MockTensorIOMode(Enum):
    """テスト用の TensorIOMode モック."""

    INPUT = 0
    OUTPUT = 1


class _StubContext:
    """execute_async_v3 の呼び出しを記録するスタブ."""

    def __init__(self) -> None:
        """スタブを初期化する.

        Attributes:
            called_with: 直近の execute_async_v3 呼び出しで受け取った
                ストリームハンドル.
        """
        self.called_with: int | None = None

    def execute_async_v3(self, stream_handle: int) -> None:
        """ストリームハンドルを記録する.

        Args:
            stream_handle: TensorRT が渡す CUDA ストリームハンドル.
        """
        self.called_with = stream_handle


class _StubStream:
    """cuda_stream 属性のみを持つスタブ."""

    def __init__(self, cuda_stream: int = 12345) -> None:
        """スタブを初期化する.

        Args:
            cuda_stream: TensorRTInference.execute に渡される
                CUDA ストリームハンドル.
        """
        self.cuda_stream = cuda_stream


class _StubEngine:
    """_resolve_io_bindings 用の最小エンジンスタブ."""

    def __init__(self, tensor_specs: list[tuple[str, _MockTensorIOMode]]) -> None:
        """スタブを初期化する.

        Args:
            tensor_specs: (tensor_name, tensor_mode) の組を並べた I/O 定義.
        """
        self._tensor_specs = list(tensor_specs)
        self.num_io_tensors = len(self._tensor_specs)
        self._mode_by_name = {name: mode for name, mode in self._tensor_specs}

    def get_tensor_name(self, index: int) -> str:
        """指定インデックスのテンソル名を返す.

        Args:
            index: I/O テンソルのインデックス.

        Returns:
            テンソル名.
        """
        return self._tensor_specs[index][0]

    def get_tensor_mode(self, name: str) -> _MockTensorIOMode:
        """指定テンソル名の I/O モードを返す.

        Args:
            name: テンソル名.

        Returns:
            対応する I/O モード.
        """
        return self._mode_by_name[name]


class TestExecute:
    """execute メソッドのテスト (TensorRT 不要)."""

    def test_calls_execute_async_v3_with_stream(self) -> None:
        """execute が stream ハンドル付きで実行されることを確認する."""
        instance = object.__new__(TensorRTInference)
        instance.context = _StubContext()
        setattr(instance, "_stream", _StubStream(12345))

        instance.execute()

        assert instance.context.called_with == 12345


class TestStreamProperty:
    """stream プロパティのテスト (TensorRT 不要)."""

    def test_stream_returns_internal_stream(self) -> None:
        """stream プロパティが内部 stream を返すことを確認する."""
        instance = object.__new__(TensorRTInference)
        stub_stream = _StubStream()
        setattr(instance, "_stream", stub_stream)

        assert instance.stream is stub_stream

    def test_stream_is_readonly(self) -> None:
        """stream プロパティが読み取り専用であることを確認する."""
        instance = object.__new__(TensorRTInference)
        setattr(instance, "_stream", _StubStream())

        with pytest.raises(AttributeError):
            setattr(instance, "stream", _StubStream())


class TestResolveIoBindings:
    """_resolve_io_bindings メソッドのテスト (TensorRT 不要)."""

    def _create_inference_with_engine(
        self,
        tensor_specs: list[tuple[str, _MockTensorIOMode]],
    ) -> TensorRTInference:
        """スタブエンジン付きの TensorRTInference インスタンスを作る.

        Args:
            tensor_specs: (tensor_name, tensor_mode) の組を並べた I/O 定義.

        Returns:
            最小構成で生成した TensorRTInference インスタンス.
        """
        instance = object.__new__(TensorRTInference)
        instance.engine = _StubEngine(tensor_specs)
        return instance

    def test_single_input_output(self) -> None:
        """入力1つ, 出力1つで正しく解決できることを確認する."""
        trt_mock = SimpleNamespace(TensorIOMode=_MockTensorIOMode)
        instance = self._create_inference_with_engine(
            [
                ("input", _MockTensorIOMode.INPUT),
                ("output", _MockTensorIOMode.OUTPUT),
            ]
        )

        result = instance._resolve_io_bindings(trt_mock)

        assert result == {"input": "input", "output": "output"}

    def test_reversed_order(self) -> None:
        """出力が先, 入力が後ろでも正しく解決できることを確認する."""
        trt_mock = SimpleNamespace(TensorIOMode=_MockTensorIOMode)
        instance = self._create_inference_with_engine(
            [
                ("output", _MockTensorIOMode.OUTPUT),
                ("input", _MockTensorIOMode.INPUT),
            ]
        )

        result = instance._resolve_io_bindings(trt_mock)

        assert result == {"input": "input", "output": "output"}

    def test_multiple_inputs_raises(self) -> None:
        """入力が複数ある場合は RuntimeError になることを確認する."""
        trt_mock = SimpleNamespace(TensorIOMode=_MockTensorIOMode)
        instance = self._create_inference_with_engine(
            [
                ("input_0", _MockTensorIOMode.INPUT),
                ("input_1", _MockTensorIOMode.INPUT),
                ("output", _MockTensorIOMode.OUTPUT),
            ]
        )

        with pytest.raises(RuntimeError, match="入力"):
            instance._resolve_io_bindings(trt_mock)

    def test_multiple_outputs_raises(self) -> None:
        """出力が複数ある場合は RuntimeError になることを確認する."""
        trt_mock = SimpleNamespace(TensorIOMode=_MockTensorIOMode)
        instance = self._create_inference_with_engine(
            [
                ("input", _MockTensorIOMode.INPUT),
                ("boxes", _MockTensorIOMode.OUTPUT),
                ("scores", _MockTensorIOMode.OUTPUT),
            ]
        )

        with pytest.raises(RuntimeError, match="出力"):
            instance._resolve_io_bindings(trt_mock)

    def test_no_input_raises(self) -> None:
        """入力が0件の場合は RuntimeError になることを確認する."""
        trt_mock = SimpleNamespace(TensorIOMode=_MockTensorIOMode)
        instance = self._create_inference_with_engine(
            [
                ("output", _MockTensorIOMode.OUTPUT),
            ]
        )

        with pytest.raises(RuntimeError, match="入力"):
            instance._resolve_io_bindings(trt_mock)

    def test_no_output_raises(self) -> None:
        """出力が0件の場合は RuntimeError になることを確認する."""
        trt_mock = SimpleNamespace(TensorIOMode=_MockTensorIOMode)
        instance = self._create_inference_with_engine(
            [
                ("input", _MockTensorIOMode.INPUT),
            ]
        )

        with pytest.raises(RuntimeError, match="出力"):
            instance._resolve_io_bindings(trt_mock)


tensorrt_available = check_tensorrt_availability()


@pytest.mark.skipif(not tensorrt_available, reason="TensorRT is not installed")
class TestTensorRTInferenceInit:
    """TensorRTInference 初期化のテスト (TensorRT 利用可能環境のみ)."""

    def test_nonexistent_engine_raises(self, tmp_path: Path) -> None:
        """存在しないエンジンファイルで FileNotFoundError になることを確認する."""
        with pytest.raises(FileNotFoundError, match="エンジンファイル"):
            TensorRTInference(tmp_path / "nonexistent.engine")

    def test_invalid_engine_raises(self, tmp_path: Path) -> None:
        """不正なエンジンファイルで RuntimeError になることを確認する."""
        invalid_engine = tmp_path / "invalid.engine"
        invalid_engine.write_bytes(b"invalid engine data")

        with pytest.raises(RuntimeError):
            TensorRTInference(invalid_engine)


@pytest.mark.skipif(tensorrt_available, reason="Test for environments without TensorRT")
class TestTensorRTInferenceWithoutTensorRT:
    """TensorRT 未インストール環境でのテスト."""

    def test_import_raises_without_tensorrt(self, tmp_path: Path) -> None:
        """TensorRT がない環境で ImportError になることを確認する."""
        dummy_engine = tmp_path / "dummy.engine"
        dummy_engine.write_bytes(b"dummy")

        with pytest.raises(ImportError, match="TensorRT"):
            TensorRTInference(dummy_engine)
