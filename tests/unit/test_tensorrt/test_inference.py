"""TensorRT推論モジュールのテスト.

TensorRTはオプション依存のため、利用不可環境ではスキップ.
check_tensorrt_availability関数はTensorRT不要でテスト可能.
"""

from pathlib import Path

import pytest

from pochitrain.tensorrt.inference import check_tensorrt_availability


class TestCheckTensorrtAvailability:
    """check_tensorrt_availability関数のテスト."""

    def test_returns_bool(self):
        """戻り値がboolであることを確認."""
        result = check_tensorrt_availability()
        assert isinstance(result, bool)


# TensorRTが利用可能な場合のみ実行するテスト
tensorrt_available = check_tensorrt_availability()


@pytest.mark.skipif(not tensorrt_available, reason="TensorRT is not installed")
class TestTensorRTInferenceInit:
    """TensorRTInference初期化のテスト（TensorRT環境のみ）."""

    def test_nonexistent_engine_raises(self, tmp_path):
        """存在しないエンジンファイルでFileNotFoundErrorが発生する."""
        from pochitrain.tensorrt.inference import TensorRTInference

        with pytest.raises(FileNotFoundError, match="エンジンファイルが見つかりません"):
            TensorRTInference(tmp_path / "nonexistent.engine")

    def test_invalid_engine_raises(self, tmp_path):
        """無効なエンジンファイルでRuntimeErrorが発生する."""
        from pochitrain.tensorrt.inference import TensorRTInference

        # 無効なエンジンファイルを作成
        invalid_engine = tmp_path / "invalid.engine"
        invalid_engine.write_bytes(b"invalid engine data")

        with pytest.raises(RuntimeError):
            TensorRTInference(invalid_engine)


@pytest.mark.skipif(tensorrt_available, reason="Test for environments without TensorRT")
class TestTensorRTInferenceWithoutTensorRT:
    """TensorRT未インストール環境でのテスト."""

    def test_import_raises_without_tensorrt(self, tmp_path):
        """TensorRTがない環境でImportErrorが発生する."""
        from pochitrain.tensorrt.inference import TensorRTInference

        dummy_engine = tmp_path / "dummy.engine"
        dummy_engine.write_bytes(b"dummy")

        with pytest.raises(ImportError, match="TensorRTがインストールされていません"):
            TensorRTInference(dummy_engine)
