"""
BaseCSVExporterのテスト.
"""

import logging
from pathlib import Path

from pochitrain.exporters import BaseCSVExporter


class ConcreteCSVExporter(BaseCSVExporter):
    """テスト用の具象クラス."""

    pass


class TestBaseCSVExporter:
    """BaseCSVExporterクラスのテスト."""

    def test_init_with_string_output_dir(self, tmp_path):
        """文字列の出力ディレクトリで初期化できること."""
        exporter = ConcreteCSVExporter(output_dir=str(tmp_path))

        assert exporter.output_dir == tmp_path
        assert exporter.logger is not None
        assert exporter.logger.name == "ConcreteCSVExporter"

    def test_init_with_path_output_dir(self, tmp_path):
        """Pathオブジェクトの出力ディレクトリで初期化できること."""
        exporter = ConcreteCSVExporter(output_dir=tmp_path)

        assert exporter.output_dir == tmp_path

    def test_init_with_custom_logger(self, tmp_path):
        """カスタムロガーを指定して初期化できること."""
        custom_logger = logging.getLogger("custom_test_logger")

        exporter = ConcreteCSVExporter(output_dir=str(tmp_path), logger=custom_logger)

        assert exporter.logger is custom_logger

    def test_generate_filename_with_none(self, tmp_path):
        """filenameがNoneの場合にタイムスタンプ付きファイル名が生成されること."""
        exporter = ConcreteCSVExporter(output_dir=str(tmp_path))

        filename = exporter._generate_filename("test_prefix")

        assert filename.startswith("test_prefix_")
        assert filename.endswith(".csv")

    def test_generate_filename_with_explicit_name(self, tmp_path):
        """明示的なファイル名が指定された場合にそのまま使用されること."""
        exporter = ConcreteCSVExporter(output_dir=str(tmp_path))

        filename = exporter._generate_filename("prefix", "my_file.csv")

        assert filename == "my_file.csv"

    def test_generate_filename_appends_csv_extension(self, tmp_path):
        """.csv拡張子がない場合に自動付与されること."""
        exporter = ConcreteCSVExporter(output_dir=str(tmp_path))

        filename = exporter._generate_filename("prefix", "my_file")

        assert filename == "my_file.csv"

    def test_build_output_path(self, tmp_path):
        """出力パスが正しく構築されること."""
        exporter = ConcreteCSVExporter(output_dir=str(tmp_path))

        path = exporter._build_output_path("test.csv")

        assert path == tmp_path / "test.csv"
