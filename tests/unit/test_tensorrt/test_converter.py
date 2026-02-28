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

    @pytest.mark.parametrize(
        "shape, input_shape, expected",
        [
            pytest.param(
                (-1, 3, -1, -1), (3, 224, 224), (1, 3, 224, 224), id="all-dynamic"
            ),
            pytest.param(
                (-1, 3, -1, -1), (3, 320, 640), (1, 3, 320, 640), id="non-square"
            ),
            pytest.param(
                (-1, 3, 224, 224),
                (3, 224, 224),
                (1, 3, 224, 224),
                id="batch-only-with-input",
            ),
            pytest.param(
                (-1, 3, 224, 224), None, (1, 3, 224, 224), id="batch-only-without-input"
            ),
            pytest.param(
                (1, 3, -1, -1),
                (3, 256, 256),
                (1, 3, 256, 256),
                id="static-batch-dynamic-spatial",
            ),
            pytest.param(
                (1, 3, 224, 224), (3, 224, 224), (1, 3, 224, 224), id="fully-static"
            ),
        ],
    )
    def test_resolves_shape(self, shape, input_shape, expected) -> None:
        """動的次元が正しく解決されることを確認する."""
        result = TensorRTConverter._resolve_dynamic_shape(shape, input_shape)
        assert result == expected

    def test_spatial_dynamic_without_input_shape_raises(self) -> None:
        """空間次元が動的でinput_shapeなしの場合, ValueErrorが発生する."""
        shape = (-1, 3, -1, -1)
        with pytest.raises(ValueError, match="input_shapeが指定されていません"):
            TensorRTConverter._resolve_dynamic_shape(shape, None)
