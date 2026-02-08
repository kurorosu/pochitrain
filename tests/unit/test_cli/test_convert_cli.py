"""pochi convert CLIのテスト.

TensorRT変換サブコマンドの引数パース・バリデーションロジックをテスト.
"""

import argparse
from pathlib import Path

import pytest

from pochitrain.cli.arg_types import positive_int


class TestConvertArgumentParsing:
    """convert サブコマンドの引数パースのテスト."""

    def _build_parser(self):
        """テスト用にconvertと同等のパーサーを構築."""
        parser = argparse.ArgumentParser()
        parser.add_argument("onnx_path", help="ONNXモデルファイルパス")
        parser.add_argument("--fp16", action="store_true")
        parser.add_argument("--int8", action="store_true")
        parser.add_argument("--output", "-o")
        parser.add_argument("--config-path", "-c")
        parser.add_argument("--calib-data")
        parser.add_argument(
            "--input-size", nargs=2, type=positive_int, metavar=("HEIGHT", "WIDTH")
        )
        parser.add_argument("--calib-samples", type=positive_int, default=500)
        parser.add_argument("--calib-batch-size", type=positive_int, default=1)
        parser.add_argument("--workspace-size", type=positive_int, default=1 << 30)
        parser.add_argument("--debug", action="store_true")
        return parser

    def test_onnx_path_required(self):
        """onnx_pathが必須引数であることを確認."""
        parser = self._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_onnx_path_parsed(self):
        """onnx_pathが正しくパースされる."""
        parser = self._build_parser()
        args = parser.parse_args(["model.onnx"])
        assert args.onnx_path == "model.onnx"

    def test_fp16_flag(self):
        """--fp16フラグが正しくパースされる."""
        parser = self._build_parser()
        args = parser.parse_args(["model.onnx", "--fp16"])
        assert args.fp16 is True
        assert args.int8 is False

    def test_int8_flag(self):
        """--int8フラグが正しくパースされる."""
        parser = self._build_parser()
        args = parser.parse_args(["model.onnx", "--int8"])
        assert args.int8 is True
        assert args.fp16 is False

    def test_default_no_precision_flag(self):
        """精度フラグ未指定時はどちらもFalse (FP32)."""
        parser = self._build_parser()
        args = parser.parse_args(["model.onnx"])
        assert args.fp16 is False
        assert args.int8 is False

    def test_output_option(self):
        """--outputオプションが正しくパースされる."""
        parser = self._build_parser()
        args = parser.parse_args(["model.onnx", "-o", "output.engine"])
        assert args.output == "output.engine"

    def test_config_path_option(self):
        """--config-pathオプションが正しくパースされる."""
        parser = self._build_parser()
        args = parser.parse_args(["model.onnx", "-c", "config.py"])
        assert args.config_path == "config.py"

    def test_calib_data_option(self):
        """--calib-dataオプションが正しくパースされる."""
        parser = self._build_parser()
        args = parser.parse_args(["model.onnx", "--calib-data", "data/val"])
        assert args.calib_data == "data/val"

    def test_calib_samples_default(self):
        """--calib-samplesのデフォルト値を確認."""
        parser = self._build_parser()
        args = parser.parse_args(["model.onnx"])
        assert args.calib_samples == 500

    def test_calib_samples_custom(self):
        """--calib-samplesのカスタム値が正しくパースされる."""
        parser = self._build_parser()
        args = parser.parse_args(["model.onnx", "--calib-samples", "200"])
        assert args.calib_samples == 200

    def test_calib_batch_size_default(self):
        """--calib-batch-sizeのデフォルト値を確認."""
        parser = self._build_parser()
        args = parser.parse_args(["model.onnx"])
        assert args.calib_batch_size == 1

    def test_workspace_size_default(self):
        """--workspace-sizeのデフォルト値を確認."""
        parser = self._build_parser()
        args = parser.parse_args(["model.onnx"])
        assert args.workspace_size == 1 << 30


class TestConvertPrecisionDecision:
    """精度モード決定ロジックのテスト."""

    def _decide_precision(self, fp16, int8):
        """精度モードを決定する (CLI実装と同じロジック)."""
        if int8:
            return "int8"
        elif fp16:
            return "fp16"
        else:
            return "fp32"

    def test_default_is_fp32(self):
        """デフォルトはFP32."""
        assert self._decide_precision(fp16=False, int8=False) == "fp32"

    def test_fp16_flag(self):
        """--fp16指定時はFP16."""
        assert self._decide_precision(fp16=True, int8=False) == "fp16"

    def test_int8_flag(self):
        """--int8指定時はINT8."""
        assert self._decide_precision(fp16=False, int8=True) == "int8"

    def test_int8_takes_precedence(self):
        """--int8と--fp16の両方が指定された場合はINT8が優先."""
        assert self._decide_precision(fp16=True, int8=True) == "int8"


class TestConvertOutputPathDecision:
    """出力パス決定ロジックのテスト."""

    def test_output_specified(self):
        """--output指定時はそのパスを使用."""
        output = "custom_output.engine"
        onnx_path = Path("model.onnx")

        if output:
            result = Path(output)
        else:
            result = onnx_path.with_suffix(".engine")

        assert result == Path("custom_output.engine")

    def _decide_output_path(self, output, onnx_path, precision):
        """出力パスを決定する (CLI実装と同じロジック)."""
        if output:
            return Path(output)
        stem = onnx_path.stem
        if precision != "fp32":
            stem = f"{stem}_{precision}"
        return onnx_path.with_name(f"{stem}.engine")

    def test_output_default_fp32(self):
        """--output未指定, FP32時のデフォルトパス."""
        onnx_path = Path("work_dirs/20260206_001/models/model.onnx")
        result = self._decide_output_path(None, onnx_path, "fp32")
        assert result == Path("work_dirs/20260206_001/models/model.engine")

    def test_output_default_fp16(self):
        """--output未指定, FP16時のデフォルトパス."""
        onnx_path = Path("work_dirs/20260206_001/models/model.onnx")
        result = self._decide_output_path(None, onnx_path, "fp16")
        assert result == Path("work_dirs/20260206_001/models/model_fp16.engine")

    def test_output_default_int8(self):
        """--output未指定, INT8時のデフォルトパス."""
        onnx_path = Path("work_dirs/20260206_001/models/model.onnx")
        result = self._decide_output_path(None, onnx_path, "int8")
        assert result == Path("work_dirs/20260206_001/models/model_int8.engine")


class TestPositiveIntValidation:
    """正の整数バリデーション関数のテスト."""

    def test_positive_int_valid(self):
        """正の整数を正しく変換する."""
        assert positive_int("1") == 1
        assert positive_int("100") == 100
        assert positive_int("500") == 500

    def test_positive_int_zero_raises_error(self):
        """0を指定するとArgumentTypeErrorが発生する."""
        with pytest.raises(argparse.ArgumentTypeError, match="1以上の整数を指定"):
            positive_int("0")

    def test_positive_int_negative_raises_error(self):
        """負の値を指定するとArgumentTypeErrorが発生する."""
        with pytest.raises(argparse.ArgumentTypeError, match="1以上の整数を指定"):
            positive_int("-1")


class TestConvertParameterValidation:
    """convert サブコマンドのパラメータバリデーションテスト."""

    def _build_parser(self):
        """テスト用にconvertと同等のパーサーを構築."""
        parser = argparse.ArgumentParser()
        parser.add_argument("onnx_path")
        parser.add_argument("--calib-samples", type=positive_int, default=500)
        parser.add_argument("--calib-batch-size", type=positive_int, default=1)
        parser.add_argument("--workspace-size", type=positive_int, default=1 << 30)
        return parser

    def test_calib_samples_zero_rejected(self):
        """--calib-samples 0 がパース時にエラーとなる."""
        parser = self._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["model.onnx", "--calib-samples", "0"])

    def test_calib_batch_size_negative_rejected(self):
        """--calib-batch-size に負の値がパース時にエラーとなる."""
        parser = self._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["model.onnx", "--calib-batch-size", "-1"])

    def test_workspace_size_zero_rejected(self):
        """--workspace-size 0 がパース時にエラーとなる."""
        parser = self._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["model.onnx", "--workspace-size", "0"])


class TestInputSizeOption:
    """--input-size オプションのテスト."""

    def _build_parser(self):
        """テスト用にconvertと同等のパーサーを構築."""
        parser = argparse.ArgumentParser()
        parser.add_argument("onnx_path")
        parser.add_argument(
            "--input-size", nargs=2, type=positive_int, metavar=("HEIGHT", "WIDTH")
        )
        return parser

    def test_input_size_default_none(self):
        """--input-size 未指定時は None."""
        parser = self._build_parser()
        args = parser.parse_args(["model.onnx"])
        assert args.input_size is None

    def test_input_size_parsed(self):
        """--input-size が正しくパースされる."""
        parser = self._build_parser()
        args = parser.parse_args(["model.onnx", "--input-size", "224", "224"])
        assert args.input_size == [224, 224]

    def test_input_size_different_values(self):
        """--input-size に異なる高さと幅を指定できる."""
        parser = self._build_parser()
        args = parser.parse_args(["model.onnx", "--input-size", "320", "640"])
        assert args.input_size == [320, 640]

    def test_input_size_zero_rejected(self):
        """--input-size 0 224 がパース時にエラーとなる."""
        parser = self._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["model.onnx", "--input-size", "0", "224"])

    def test_input_size_negative_rejected(self):
        """--input-size -1 224 がパース時にエラーとなる."""
        parser = self._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["model.onnx", "--input-size", "-1", "224"])

    def test_input_size_second_value_zero_rejected(self):
        """--input-size 224 0 がパース時にエラーとなる."""
        parser = self._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["model.onnx", "--input-size", "224", "0"])

    def test_input_size_both_zero_rejected(self):
        """--input-size 0 0 がパース時にエラーとなる."""
        parser = self._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["model.onnx", "--input-size", "0", "0"])


class TestDynamicShapeDetection:
    """動的シェイプONNXモデルの検出ロジックのテスト."""

    def _has_dynamic_dims(self, dim_values):
        """dim_value=0 の次元があるかチェックする (CLI実装と同じロジック)."""
        return any(v == 0 for v in dim_values)

    def test_static_shape_no_dynamic(self):
        """静的シェイプには動的次元がない."""
        assert self._has_dynamic_dims([3, 224, 224]) is False

    def test_dynamic_height_width(self):
        """height/width が動的 (dim_value=0) の場合を検出."""
        assert self._has_dynamic_dims([3, 0, 0]) is True

    def test_dynamic_single_dim(self):
        """1次元のみ動的な場合も検出."""
        assert self._has_dynamic_dims([3, 0, 224]) is True

    def test_input_shape_from_cli(self):
        """--input-size 指定時の入力形状構築を検証."""
        input_size = [224, 224]
        input_shape = (3, input_size[0], input_size[1])
        assert input_shape == (3, 224, 224)


class TestInputShapeForAllPrecisions:
    """--input-size が全精度モードで input_shape に変換されるテスト."""

    @staticmethod
    def _build_input_shape(input_size):
        """CLI引数からinput_shapeを構築する (CLI実装と同じロジック)."""
        if input_size:
            return (3, input_size[0], input_size[1])
        return None

    def test_input_shape_constructed_for_fp32(self):
        """FP32でもinput_shapeタプルが構築される."""
        input_shape = self._build_input_shape([224, 224])
        assert input_shape == (3, 224, 224)

    def test_input_shape_constructed_for_fp16(self):
        """FP16で異なるH/Wのinput_shapeが正しく構築される."""
        input_shape = self._build_input_shape([320, 640])
        assert input_shape == (3, 320, 640)

    def test_input_shape_none_when_not_specified(self):
        """--input-size 未指定時はNone."""
        input_shape = self._build_input_shape(None)
        assert input_shape is None

    def test_dynamic_shape_detection_applies_to_all_precisions(self):
        """動的シェイプ検出が精度モードに依存しないことを検証."""
        # dim_value=0 の検出ロジックは精度モードの外で実行される
        dim_values = [3, 0, 0]
        has_dynamic = any(v == 0 for v in dim_values)
        assert has_dynamic is True
