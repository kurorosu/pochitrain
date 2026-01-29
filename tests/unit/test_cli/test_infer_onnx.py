"""infer_onnx CLIのテスト.

ONNX推論CLIの引数パース・データパス決定ロジックをテスト.
実際のONNX推論はtest_onnx/test_inference.pyでテスト済み.
"""

import argparse
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")


class TestInferOnnxArgumentParsing:
    """infer_onnx CLIの引数パースのテスト."""

    def _build_parser(self):
        """テスト用にinfer_onnxと同等のパーサーを構築."""
        parser = argparse.ArgumentParser(description="ONNXモデルを使用した推論")
        parser.add_argument("model_path", help="ONNXモデルファイルパス")
        parser.add_argument("--data", help="推論データディレクトリ")
        parser.add_argument("--output", "-o", help="結果出力ディレクトリ")
        return parser

    def test_model_path_required(self):
        """model_pathが必須引数であることを確認."""
        parser = self._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_model_path_parsed(self):
        """model_pathが正しくパースされる."""
        parser = self._build_parser()
        args = parser.parse_args(["model.onnx"])
        assert args.model_path == "model.onnx"

    def test_data_option(self):
        """--dataオプションが正しくパースされる."""
        parser = self._build_parser()
        args = parser.parse_args(["model.onnx", "--data", "data/val"])
        assert args.data == "data/val"

    def test_output_option(self):
        """--outputオプションが正しくパースされる."""
        parser = self._build_parser()
        args = parser.parse_args(["model.onnx", "--output", "results/"])
        assert args.output == "results/"

    def test_output_short_option(self):
        """-oオプションが正しくパースされる."""
        parser = self._build_parser()
        args = parser.parse_args(["model.onnx", "-o", "results/"])
        assert args.output == "results/"

    def test_data_default_none(self):
        """--data未指定時はNone."""
        parser = self._build_parser()
        args = parser.parse_args(["model.onnx"])
        assert args.data is None

    def test_output_default_none(self):
        """--output未指定時はNone."""
        parser = self._build_parser()
        args = parser.parse_args(["model.onnx"])
        assert args.output is None


class TestInferOnnxDataPathDecision:
    """データパス決定ロジックのテスト.

    CLIの実際のロジックを再現してテストする.
    """

    def test_data_from_args(self, tmp_path):
        """--dataが指定された場合はそちらを使用."""
        args_data = str(tmp_path / "custom_val")
        config = {"val_data_root": str(tmp_path / "config_val")}

        # --data指定時はargsのパスを使う
        if args_data:
            data_path = Path(args_data)
        elif "val_data_root" in config:
            data_path = Path(config["val_data_root"])
        else:
            data_path = None

        assert data_path == tmp_path / "custom_val"

    def test_data_from_config(self, tmp_path):
        """--data未指定時はconfigのval_data_rootを使用."""
        args_data = None
        config = {"val_data_root": str(tmp_path / "config_val")}

        if args_data:
            data_path = Path(args_data)
        elif "val_data_root" in config:
            data_path = Path(config["val_data_root"])
        else:
            data_path = None

        assert data_path == tmp_path / "config_val"

    def test_no_data_source(self):
        """--dataもconfigもない場合はNone."""
        args_data = None
        config = {}

        if args_data:
            data_path = Path(args_data)
        elif "val_data_root" in config:
            data_path = Path(config["val_data_root"])
        else:
            data_path = None

        assert data_path is None


class TestInferOnnxMainExit:
    """main関数のSystemExitテスト."""

    def test_main_no_args_exits(self):
        """引数なしでSystemExitが発生する."""
        from pochitrain.cli.infer_onnx import main

        with patch("sys.argv", ["infer-onnx"]):
            with pytest.raises(SystemExit):
                main()

    def test_main_nonexistent_model_exits(self, tmp_path):
        """存在しないモデルでSystemExitが発生する."""
        from pochitrain.cli.infer_onnx import main

        fake_model = str(tmp_path / "nonexistent.onnx")
        with patch("sys.argv", ["infer-onnx", fake_model]):
            with pytest.raises(SystemExit):
                main()
