"""export-onnx CLIのテスト.

--input-size バリデーションを中心にテスト.
"""

import argparse

import pytest

from pochitrain.cli.arg_types import positive_int


class TestExportOnnxArgumentParsing:
    """export-onnx コマンドの引数パースのテスト."""

    def _build_parser(self):
        """テスト用にexport-onnxと同等のパーサーを構築."""
        parser = argparse.ArgumentParser()
        parser.add_argument("model_path", help="変換するPyTorchモデルファイル(.pth)")
        parser.add_argument("--config", "-c")
        parser.add_argument("--model-name", default="resnet18")
        parser.add_argument("--num-classes", type=int)
        parser.add_argument(
            "--input-size",
            nargs=2,
            type=positive_int,
            required=True,
            metavar=("HEIGHT", "WIDTH"),
        )
        parser.add_argument("--output", "-o")
        parser.add_argument("--opset-version", type=int, default=17)
        parser.add_argument("--device", default="cpu")
        parser.add_argument("--skip-verify", action="store_true")
        return parser

    def test_model_path_required(self):
        """model_pathが必須引数であることを確認."""
        parser = self._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--input-size", "224", "224"])

    def test_input_size_required(self):
        """--input-sizeが必須引数であることを確認."""
        parser = self._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["model.pth"])

    def test_valid_input_size_parsed(self):
        """有効な--input-sizeが正しくパースされる."""
        parser = self._build_parser()
        args = parser.parse_args(["model.pth", "--input-size", "224", "224"])
        assert args.input_size == [224, 224]

    def test_input_size_different_values(self):
        """異なる高さと幅を指定できる."""
        parser = self._build_parser()
        args = parser.parse_args(["model.pth", "--input-size", "320", "640"])
        assert args.input_size == [320, 640]

    def test_default_values(self):
        """デフォルト値が正しく設定される."""
        parser = self._build_parser()
        args = parser.parse_args(["model.pth", "--input-size", "224", "224"])
        assert args.model_name == "resnet18"
        assert args.opset_version == 17
        assert args.device == "cpu"
        assert args.skip_verify is False
        assert args.output is None
        assert args.config is None
        assert args.num_classes is None


class TestExportOnnxInputSizeValidation:
    """export-onnx の --input-size バリデーションテスト."""

    def _build_parser(self):
        """テスト用にexport-onnxと同等のパーサーを構築."""
        parser = argparse.ArgumentParser()
        parser.add_argument("model_path")
        parser.add_argument(
            "--input-size",
            nargs=2,
            type=positive_int,
            required=True,
            metavar=("HEIGHT", "WIDTH"),
        )
        return parser

    def test_input_size_zero_zero_rejected(self):
        """--input-size 0 0 がパース時にエラーとなる."""
        parser = self._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["model.pth", "--input-size", "0", "0"])

    def test_input_size_negative_negative_rejected(self):
        """--input-size -1 -1 がパース時にエラーとなる."""
        parser = self._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["model.pth", "--input-size", "-1", "-1"])

    def test_input_size_first_value_zero_rejected(self):
        """--input-size 0 224 がパース時にエラーとなる."""
        parser = self._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["model.pth", "--input-size", "0", "224"])

    def test_input_size_second_value_zero_rejected(self):
        """--input-size 224 0 がパース時にエラーとなる."""
        parser = self._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["model.pth", "--input-size", "224", "0"])

    def test_input_size_first_value_negative_rejected(self):
        """--input-size -1 224 がパース時にエラーとなる."""
        parser = self._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["model.pth", "--input-size", "-1", "224"])

    def test_input_size_second_value_negative_rejected(self):
        """--input-size 224 -1 がパース時にエラーとなる."""
        parser = self._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["model.pth", "--input-size", "224", "-1"])
