"""TensorRT推論モジュールのテスト.

TensorRTはオプション依存のため, 利用不可環境ではスキップ.
check_tensorrt_availability関数とI/Oバインディング検証ロジックは
TensorRT不要でテスト可能.
"""

from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from pochitrain.tensorrt.inference import TensorRTInference, check_tensorrt_availability


class TestCheckTensorrtAvailability:
    """check_tensorrt_availability関数のテスト."""

    def test_returns_bool(self):
        """戻り値がboolであることを確認."""
        result = check_tensorrt_availability()
        assert isinstance(result, bool)


class _MockTensorIOMode(Enum):
    """テスト用のTensorIOModeモック."""

    INPUT = 0
    OUTPUT = 1


class TestResolveIoBindings:
    """_resolve_io_bindings メソッドの単体テスト (TensorRT不要)."""

    def _create_inference_with_mock_engine(self, tensor_specs):
        """モックエンジンでTensorRTInferenceインスタンスを部分的に構築する.

        Args:
            tensor_specs: [(name, mode), ...] のリスト.
                mode は _MockTensorIOMode.INPUT または OUTPUT.

        Returns:
            engine属性のみ設定済みのTensorRTInferenceインスタンス
        """
        mock_engine = MagicMock()
        mock_engine.num_io_tensors = len(tensor_specs)
        mock_engine.get_tensor_name.side_effect = [s[0] for s in tensor_specs]
        mock_engine.get_tensor_mode.side_effect = [s[1] for s in tensor_specs]

        # __init__をバイパスして直接インスタンスを作る
        instance = object.__new__(TensorRTInference)
        instance.engine = mock_engine
        return instance

    def test_single_input_output(self):
        """入力1つ・出力1つで正常に解決できる."""
        trt_mock = SimpleNamespace(TensorIOMode=_MockTensorIOMode)
        instance = self._create_inference_with_mock_engine(
            [
                ("input", _MockTensorIOMode.INPUT),
                ("output", _MockTensorIOMode.OUTPUT),
            ]
        )

        result = instance._resolve_io_bindings(trt_mock)

        assert result == {"input": "input", "output": "output"}

    def test_reversed_order(self):
        """出力が先, 入力が後の順序でも正しく解決できる."""
        trt_mock = SimpleNamespace(TensorIOMode=_MockTensorIOMode)
        instance = self._create_inference_with_mock_engine(
            [
                ("output", _MockTensorIOMode.OUTPUT),
                ("input", _MockTensorIOMode.INPUT),
            ]
        )

        result = instance._resolve_io_bindings(trt_mock)

        assert result == {"input": "input", "output": "output"}

    def test_multiple_inputs_raises(self):
        """入力が2つ以上の場合はRuntimeErrorが発生する."""
        trt_mock = SimpleNamespace(TensorIOMode=_MockTensorIOMode)
        instance = self._create_inference_with_mock_engine(
            [
                ("input_0", _MockTensorIOMode.INPUT),
                ("input_1", _MockTensorIOMode.INPUT),
                ("output", _MockTensorIOMode.OUTPUT),
            ]
        )

        with pytest.raises(RuntimeError, match="単一入力のみサポート"):
            instance._resolve_io_bindings(trt_mock)

    def test_multiple_outputs_raises(self):
        """出力が2つ以上の場合はRuntimeErrorが発生する."""
        trt_mock = SimpleNamespace(TensorIOMode=_MockTensorIOMode)
        instance = self._create_inference_with_mock_engine(
            [
                ("input", _MockTensorIOMode.INPUT),
                ("boxes", _MockTensorIOMode.OUTPUT),
                ("scores", _MockTensorIOMode.OUTPUT),
            ]
        )

        with pytest.raises(RuntimeError, match="単一出力のみサポート"):
            instance._resolve_io_bindings(trt_mock)

    def test_no_input_raises(self):
        """入力が0個の場合はRuntimeErrorが発生する."""
        trt_mock = SimpleNamespace(TensorIOMode=_MockTensorIOMode)
        instance = self._create_inference_with_mock_engine(
            [
                ("output", _MockTensorIOMode.OUTPUT),
            ]
        )

        with pytest.raises(RuntimeError, match="単一入力のみサポート"):
            instance._resolve_io_bindings(trt_mock)

    def test_no_output_raises(self):
        """出力が0個の場合はRuntimeErrorが発生する."""
        trt_mock = SimpleNamespace(TensorIOMode=_MockTensorIOMode)
        instance = self._create_inference_with_mock_engine(
            [
                ("input", _MockTensorIOMode.INPUT),
            ]
        )

        with pytest.raises(RuntimeError, match="単一出力のみサポート"):
            instance._resolve_io_bindings(trt_mock)


# TensorRTが利用可能な場合のみ実行するテスト
tensorrt_available = check_tensorrt_availability()


@pytest.mark.skipif(not tensorrt_available, reason="TensorRT is not installed")
class TestTensorRTInferenceInit:
    """TensorRTInference初期化のテスト (TensorRT環境のみ)."""

    def test_nonexistent_engine_raises(self, tmp_path):
        """存在しないエンジンファイルでFileNotFoundErrorが発生する."""
        with pytest.raises(FileNotFoundError, match="エンジンファイルが見つかりません"):
            TensorRTInference(tmp_path / "nonexistent.engine")

    def test_invalid_engine_raises(self, tmp_path):
        """無効なエンジンファイルでRuntimeErrorが発生する."""
        invalid_engine = tmp_path / "invalid.engine"
        invalid_engine.write_bytes(b"invalid engine data")

        with pytest.raises(RuntimeError):
            TensorRTInference(invalid_engine)


@pytest.mark.skipif(tensorrt_available, reason="Test for environments without TensorRT")
class TestTensorRTInferenceWithoutTensorRT:
    """TensorRT未インストール環境でのテスト."""

    def test_import_raises_without_tensorrt(self, tmp_path):
        """TensorRTがない環境でImportErrorが発生する."""
        dummy_engine = tmp_path / "dummy.engine"
        dummy_engine.write_bytes(b"dummy")

        with pytest.raises(ImportError, match="TensorRTがインストールされていません"):
            TensorRTInference(dummy_engine)
