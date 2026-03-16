"""benchmark/utils.py のテスト."""

import json
from pathlib import Path

from pochitrain.benchmark.utils import (
    configure_logger,
    now_jst_timestamp,
    now_local_timestamp,
    to_float,
    write_json,
)


class TestConfigureLogger:
    """configure_logger のテスト."""

    def test_returns_logger_with_benchmark_name(self):
        """pochitrain.benchmark という名前のロガーを返す."""
        logger = configure_logger()
        assert logger.name == "pochitrain.benchmark"

    def test_debug_mode(self):
        """debug=True で DEBUG レベルが設定される."""
        import logging

        logger = configure_logger(debug=True)
        assert logger.level == logging.DEBUG


class TestNowJstTimestamp:
    """now_jst_timestamp のテスト."""

    def test_format(self):
        """YYYY-MM-DD HH:MM:SS 形式の文字列を返す."""
        result = now_jst_timestamp()
        # 形式: 2026-03-16 12:34:56
        assert len(result) == 19
        assert result[4] == "-"
        assert result[7] == "-"
        assert result[10] == " "
        assert result[13] == ":"
        assert result[16] == ":"


class TestNowLocalTimestamp:
    """now_local_timestamp のテスト."""

    def test_format(self):
        """YYYYMMDD_HHMMSS 形式の文字列を返す."""
        result = now_local_timestamp()
        # 形式: 20260316_123456
        assert len(result) == 15
        assert result[8] == "_"
        assert result[:8].isdigit()
        assert result[9:].isdigit()


class TestWriteJson:
    """write_json のテスト."""

    def test_writes_valid_json(self, tmp_path: Path):
        """有効な JSON ファイルを書き出す."""
        output = tmp_path / "result.json"
        payload = {"key": "value", "number": 42}
        write_json(output, payload)

        with open(output, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == payload

    def test_creates_parent_directories(self, tmp_path: Path):
        """親ディレクトリが存在しなくても作成される."""
        output = tmp_path / "nested" / "dir" / "result.json"
        write_json(output, {"a": 1})
        assert output.exists()

    def test_japanese_characters_preserved(self, tmp_path: Path):
        """日本語文字がエスケープされずに保持される."""
        output = tmp_path / "jp.json"
        write_json(output, {"名前": "テスト"})

        raw = output.read_text(encoding="utf-8")
        assert "テスト" in raw  # ensure_ascii=False


class TestToFloat:
    """to_float のテスト."""

    def test_int_conversion(self):
        """int を float に変換する."""
        assert to_float(42) == 42.0

    def test_float_passthrough(self):
        """float はそのまま返す."""
        assert to_float(3.14) == 3.14

    def test_string_number(self):
        """数値文字列を float に変換する."""
        assert to_float("10.5") == 10.5

    def test_none_returns_none(self):
        """None は None を返す."""
        assert to_float(None) is None

    def test_non_numeric_string_returns_none(self):
        """非数値文字列は None を返す."""
        assert to_float("abc") is None

    def test_list_returns_none(self):
        """リストは None を返す."""
        assert to_float([1, 2]) is None
