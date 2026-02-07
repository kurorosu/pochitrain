"""TensorRTコンバーターのテスト.

TensorRTはオプション依存のため, TensorRT不要なバリデーションテストと
TensorRT必須のテストを分離.
"""

from pathlib import Path

import pytest

from pochitrain.tensorrt.converter import TensorRTConverter
from pochitrain.tensorrt.inference import check_tensorrt_availability


class TestTensorRTConverterValidation:
    """TensorRTConverter のバリデーションテスト (TensorRT不要)."""

    def test_nonexistent_onnx_raises(self, tmp_path):
        """存在しないONNXファイルでFileNotFoundErrorが発生する."""
        if not check_tensorrt_availability():
            pytest.skip("TensorRT is not installed")

        from pochitrain.tensorrt.converter import TensorRTConverter

        with pytest.raises(FileNotFoundError, match="ONNXモデルが見つかりません"):
            TensorRTConverter(tmp_path / "nonexistent.onnx")

    def test_import_error_without_tensorrt(self, tmp_path):
        """TensorRTがない環境でImportErrorが発生する."""
        if check_tensorrt_availability():
            pytest.skip("TensorRT is installed, skipping non-TensorRT test")

        from pochitrain.tensorrt.converter import TensorRTConverter

        dummy_onnx = tmp_path / "model.onnx"
        dummy_onnx.write_bytes(b"dummy")

        with pytest.raises(ImportError, match="TensorRTがインストールされていません"):
            TensorRTConverter(dummy_onnx)


tensorrt_available = check_tensorrt_availability()


@pytest.mark.skipif(not tensorrt_available, reason="TensorRT is not installed")
class TestTensorRTConverterConvert:
    """TensorRTConverter.convert() のテスト (TensorRT環境のみ)."""

    def test_invalid_precision_raises(self, tmp_path):
        """無効な精度モードでValueErrorが発生する."""
        from pochitrain.tensorrt.converter import TensorRTConverter

        dummy_onnx = tmp_path / "model.onnx"
        dummy_onnx.write_bytes(b"dummy")

        converter = TensorRTConverter(dummy_onnx)
        with pytest.raises(ValueError, match="無効なprecision"):
            converter.convert(
                output_path=tmp_path / "model.engine",
                precision="int4",
            )

    def test_int8_without_calibrator_raises(self, tmp_path):
        """INT8でcalibratorなしの場合にValueErrorが発生する."""
        from pochitrain.tensorrt.converter import TensorRTConverter

        dummy_onnx = tmp_path / "model.onnx"
        dummy_onnx.write_bytes(b"dummy")

        converter = TensorRTConverter(dummy_onnx)
        with pytest.raises(ValueError, match="calibratorが必須"):
            converter.convert(
                output_path=tmp_path / "model.engine",
                precision="int8",
                calibrator=None,
            )


class TestResolveDynamicShape:
    """_resolve_dynamic_shape のテスト (TensorRT不要)."""

    def test_all_dynamic_with_input_shape(self):
        """全動的次元がinput_shapeで正しく解決される."""
        shape = (-1, 3, -1, -1)
        input_shape = (3, 224, 224)
        result = TensorRTConverter._resolve_dynamic_shape(shape, input_shape)
        assert result == (1, 3, 224, 224)

    def test_non_square_input_shape(self):
        """非正方形の入力サイズが正しく解決される."""
        shape = (-1, 3, -1, -1)
        input_shape = (3, 320, 640)
        result = TensorRTConverter._resolve_dynamic_shape(shape, input_shape)
        assert result == (1, 3, 320, 640)

    def test_batch_only_dynamic_with_input_shape(self):
        """バッチ次元のみ動的な場合, input_shapeありで正しく解決される."""
        shape = (-1, 3, 224, 224)
        input_shape = (3, 224, 224)
        result = TensorRTConverter._resolve_dynamic_shape(shape, input_shape)
        assert result == (1, 3, 224, 224)

    def test_batch_only_dynamic_without_input_shape(self):
        """バッチ次元のみ動的な場合, input_shapeなしでも正しく解決される."""
        shape = (-1, 3, 224, 224)
        result = TensorRTConverter._resolve_dynamic_shape(shape, None)
        assert result == (1, 3, 224, 224)

    def test_static_batch_dynamic_spatial(self):
        """静的バッチ+動的空間次元がinput_shapeで解決される."""
        shape = (1, 3, -1, -1)
        input_shape = (3, 256, 256)
        result = TensorRTConverter._resolve_dynamic_shape(shape, input_shape)
        assert result == (1, 3, 256, 256)

    def test_fully_static_shape(self):
        """完全に静的なshapeはそのまま返される."""
        shape = (1, 3, 224, 224)
        input_shape = (3, 224, 224)
        result = TensorRTConverter._resolve_dynamic_shape(shape, input_shape)
        assert result == (1, 3, 224, 224)

    def test_spatial_dynamic_without_input_shape_raises(self):
        """空間次元が動的でinput_shapeなしの場合, ValueErrorが発生する."""
        shape = (-1, 3, -1, -1)
        with pytest.raises(ValueError, match="input_shapeが指定されていません"):
            TensorRTConverter._resolve_dynamic_shape(shape, None)
