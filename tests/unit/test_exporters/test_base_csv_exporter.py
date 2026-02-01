"""
BaseCSVExporterのテスト.
"""

import tempfile
from pathlib import Path

from pochitrain.exporters import BaseCSVExporter


class ConcreteCSVExporter(BaseCSVExporter):
    """テスト用の具象クラス."""

    pass


class TestBaseCSVExporter:
    """BaseCSVExporterクラスのテスト."""

    def test_init_with_string_output_dir(self):
        """文字列の出力ディレクトリで初期化できること."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = ConcreteCSVExporter(output_dir=temp_dir)

            assert exporter.output_dir == Path(temp_dir)
            assert exporter.logger is not None
            assert exporter.logger.name == "ConcreteCSVExporter"

    def test_init_with_path_output_dir(self):
        """Pathオブジェクトの出力ディレクトリで初期化できること."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = ConcreteCSVExporter(output_dir=Path(temp_dir))

            assert exporter.output_dir == Path(temp_dir)

    def test_init_with_custom_logger(self):
        """カスタムロガーを指定して初期化できること."""
        import logging

        custom_logger = logging.getLogger("custom_test_logger")

        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = ConcreteCSVExporter(output_dir=temp_dir, logger=custom_logger)

            assert exporter.logger is custom_logger

    def test_generate_filename_with_none(self):
        """filenameがNoneの場合にタイムスタンプ付きファイル名が生成されること."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = ConcreteCSVExporter(output_dir=temp_dir)

            filename = exporter._generate_filename("test_prefix")

            assert filename.startswith("test_prefix_")
            assert filename.endswith(".csv")

    def test_generate_filename_with_explicit_name(self):
        """明示的なファイル名が指定された場合にそのまま使用されること."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = ConcreteCSVExporter(output_dir=temp_dir)

            filename = exporter._generate_filename("prefix", "my_file.csv")

            assert filename == "my_file.csv"

    def test_generate_filename_appends_csv_extension(self):
        """.csv拡張子がない場合に自動付与されること."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = ConcreteCSVExporter(output_dir=temp_dir)

            filename = exporter._generate_filename("prefix", "my_file")

            assert filename == "my_file.csv"

    def test_generate_filename_does_not_double_csv(self):
        """既に.csv拡張子がある場合に二重付与されないこと."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = ConcreteCSVExporter(output_dir=temp_dir)

            filename = exporter._generate_filename("prefix", "my_file.csv")

            assert filename == "my_file.csv"

    def test_build_output_path(self):
        """出力パスが正しく構築されること."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = ConcreteCSVExporter(output_dir=temp_dir)

            path = exporter._build_output_path("test.csv")

            assert path == Path(temp_dir) / "test.csv"

    def test_subclass_inherits_logger_name(self):
        """サブクラスのロガー名がクラス名になること."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = ConcreteCSVExporter(output_dir=temp_dir)

            assert exporter.logger.name == "ConcreteCSVExporter"
