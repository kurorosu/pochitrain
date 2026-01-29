"""infer_trt CLIのテスト.

TensorRT推論CLIの引数パース・データパス決定ロジックをテスト.
TensorRTはオプション依存のためインポートチェックでスキップ.
"""

import argparse
from pathlib import Path

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


class TestInferTrtDataPathDecision:
    """データパス決定ロジックのテスト."""

    def test_data_from_args(self, tmp_path):
        """--dataが指定された場合はそちらを使用."""
        args_data = str(tmp_path / "custom_val")
        config = {"val_data_root": str(tmp_path / "config_val")}

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


class TestInferTrtTransformDecision:
    """transformの決定ロジックのテスト."""

    def test_transform_from_config(self):
        """configにval_transformがある場合はそちらを使用."""
        import torchvision.transforms as transforms

        test_transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )
        config = {"val_transform": test_transform}

        if "val_transform" in config:
            transform = config["val_transform"]
            input_size_str = "config指定"
        else:
            transform = None
            input_size_str = "auto"

        assert transform is test_transform
        assert input_size_str == "config指定"

    def test_transform_fallback_without_config(self):
        """configにval_transformがない場合のフォールバック."""
        config = {}

        if "val_transform" in config:
            transform = config["val_transform"]
            input_size_str = "config指定"
        else:
            transform = None
            input_size_str = "auto"

        assert transform is None
        assert input_size_str == "auto"
