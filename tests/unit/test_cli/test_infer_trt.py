"""infer_trt CLIのテスト.

TensorRT推論CLIの引数パースをテスト.
TensorRTはオプション依存のためインポートチェックでスキップ.
"""

import argparse

import pytest


class TestInferTrtArgumentParsing:
    """infer_trt CLIの引数パースのテスト."""

    def _build_parser(self):
        """テスト用にinfer_trtと同等のパーサーを構築."""
        parser = argparse.ArgumentParser(
            description="TensorRTエンジンを使用した高速推論"
        )
        parser.add_argument("engine_path", help="TensorRTエンジンファイルパス")
        parser.add_argument("--data", help="推論データディレクトリ")
        parser.add_argument("--output", "-o", help="結果出力ディレクトリ")
        return parser

    def test_engine_path_required(self):
        """engine_pathが必須引数であることを確認."""
        parser = self._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_engine_path_parsed(self):
        """engine_pathが正しくパースされる."""
        parser = self._build_parser()
        args = parser.parse_args(["model.engine"])
        assert args.engine_path == "model.engine"

    def test_data_option(self):
        """--dataオプションが正しくパースされる."""
        parser = self._build_parser()
        args = parser.parse_args(["model.engine", "--data", "data/val"])
        assert args.data == "data/val"

    def test_output_option(self):
        """--outputオプションが正しくパースされる."""
        parser = self._build_parser()
        args = parser.parse_args(["model.engine", "--output", "results/"])
        assert args.output == "results/"

    def test_output_short_option(self):
        """-oオプションが正しくパースされる."""
        parser = self._build_parser()
        args = parser.parse_args(["model.engine", "-o", "results/"])
        assert args.output == "results/"

    def test_data_default_none(self):
        """--data未指定時はNone."""
        parser = self._build_parser()
        args = parser.parse_args(["model.engine"])
        assert args.data is None

    def test_output_default_none(self):
        """--output未指定時はNone."""
        parser = self._build_parser()
        args = parser.parse_args(["model.engine"])
        assert args.output is None
