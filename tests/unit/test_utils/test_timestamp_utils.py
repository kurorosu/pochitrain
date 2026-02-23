"""timestamp_utilsモジュールのテスト.

既存のtest_workspace_managerで間接的にテストされている部分もあるが、
各関数を直接テストしてカバレッジを拡充する.
"""

import re
from datetime import datetime
from pathlib import Path

import pytest

import pochitrain.utils.timestamp_utils as timestamp_utils
from pochitrain.utils.timestamp_utils import (
    find_next_index,
    format_workspace_name,
    get_current_date_str,
    get_current_timestamp,
    parse_timestamp_dir,
)


class TestFindNextIndex:
    """find_next_index関数のテスト."""

    def test_empty_dir(self, tmp_path):
        """空ディレクトリでは1を返す."""
        assert find_next_index(tmp_path, "20260129") == 1

    def test_nonexistent_dir(self, tmp_path):
        """存在しないディレクトリでは1を返す."""
        assert find_next_index(tmp_path / "nonexistent", "20260129") == 1

    def test_one_existing(self, tmp_path):
        """1つ存在する場合は2を返す."""
        (tmp_path / "20260129_001").mkdir()
        assert find_next_index(tmp_path, "20260129") == 2

    def test_multiple_existing(self, tmp_path):
        """複数存在する場合は最大+1を返す."""
        (tmp_path / "20260129_001").mkdir()
        (tmp_path / "20260129_002").mkdir()
        (tmp_path / "20260129_005").mkdir()
        assert find_next_index(tmp_path, "20260129") == 6

    def test_different_dates_ignored(self, tmp_path):
        """異なる日付のディレクトリは無視される."""
        (tmp_path / "20260128_001").mkdir()
        (tmp_path / "20260128_002").mkdir()
        assert find_next_index(tmp_path, "20260129") == 1

    def test_non_matching_dirs_ignored(self, tmp_path):
        """形式が合わないディレクトリは無視される."""
        (tmp_path / "some_other_dir").mkdir()
        (tmp_path / "20260129_abc").mkdir()
        assert find_next_index(tmp_path, "20260129") == 1

    def test_files_ignored(self, tmp_path):
        """ファイルは無視される."""
        (tmp_path / "20260129_001").touch()  # ファイル（ディレクトリではない）
        assert find_next_index(tmp_path, "20260129") == 1


class TestParseTimestampDir:
    """parse_timestamp_dir関数のテスト."""

    def test_valid_format(self):
        """正しい形式をパースできる."""
        date_str, index = parse_timestamp_dir("20241220_001")
        assert date_str == "20241220"
        assert index == 1

    def test_larger_index(self):
        """3桁インデックスをパースできる."""
        date_str, index = parse_timestamp_dir("20241220_123")
        assert date_str == "20241220"
        assert index == 123

    def test_invalid_format_raises(self):
        """不正な形式でValueErrorが発生する."""
        with pytest.raises(ValueError, match="Invalid directory name format"):
            parse_timestamp_dir("invalid_name")

    def test_too_few_digits_raises(self):
        """日付の桁数が足りない場合."""
        with pytest.raises(ValueError):
            parse_timestamp_dir("2024122_001")

    def test_too_many_index_digits_raises(self):
        """インデックスの桁数が多い場合."""
        with pytest.raises(ValueError):
            parse_timestamp_dir("20241220_0001")

    def test_no_underscore_raises(self):
        """アンダースコアがない場合."""
        with pytest.raises(ValueError):
            parse_timestamp_dir("20241220001")


class TestGetCurrentDateStr:
    """get_current_date_str関数のテスト."""

    def test_format(self):
        """yyyymmdd形式であることを確認."""
        result = get_current_date_str()
        assert re.match(r"^\d{8}$", result)

    def test_matches_today(self, monkeypatch):
        """現在日付文字列の判定が決定論的であることを確認."""

        class FixedDatetime(datetime):
            @classmethod
            def now(cls):
                return cls(2026, 2, 14, 1, 2, 3)

        monkeypatch.setattr(timestamp_utils, "datetime", FixedDatetime)

        assert get_current_date_str() == "20260214"


class TestGetCurrentTimestamp:
    """get_current_timestamp関数のテスト."""

    def test_format(self):
        """yyyymmdd_hhmmss形式であることを確認."""
        result = get_current_timestamp()
        assert re.match(r"^\d{8}_\d{6}$", result)

    def test_starts_with_today(self, monkeypatch):
        """現在時刻文字列の判定が決定論的であることを確認."""

        class FixedDatetime(datetime):
            @classmethod
            def now(cls):
                return cls(2026, 2, 14, 7, 8, 9)

        monkeypatch.setattr(timestamp_utils, "datetime", FixedDatetime)

        assert get_current_timestamp() == "20260214_070809"


class TestFormatWorkspaceName:
    """format_workspace_name関数のテスト."""

    def test_basic(self):
        """基本的なフォーマット."""
        assert format_workspace_name("20241220", 1) == "20241220_001"

    def test_larger_index(self):
        """大きなインデックス."""
        assert format_workspace_name("20241220", 42) == "20241220_042"

    def test_three_digit_index(self):
        """3桁インデックス."""
        assert format_workspace_name("20241220", 100) == "20241220_100"

    def test_roundtrip_with_parse(self):
        """format -> parse のラウンドトリップが一致する."""
        name = format_workspace_name("20260129", 7)
        date_str, index = parse_timestamp_dir(name)
        assert date_str == "20260129"
        assert index == 7
