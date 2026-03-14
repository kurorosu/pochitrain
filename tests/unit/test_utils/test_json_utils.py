"""json_utils ユーティリティのテスト."""

import json

import pytest

from pochitrain.utils.json_utils import write_json_file


class TestWriteJsonFile:
    """write_json_file のテストクラス."""

    def test_write_and_read_back(self, tmp_path):
        """書き出したJSONファイルを読み戻して内容が一致すること."""
        output_path = tmp_path / "output.json"
        payload = {"name": "test", "values": [1, 2, 3]}

        result = write_json_file(output_path, payload)

        assert result == output_path
        with open(output_path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == payload

    def test_creates_parent_directories(self, tmp_path):
        """親ディレクトリが存在しない場合に自動作成されること."""
        output_path = tmp_path / "nested" / "dir" / "output.json"
        payload = {"key": "value"}

        write_json_file(output_path, payload)

        assert output_path.exists()
        with open(output_path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == payload

    def test_japanese_not_escaped(self, tmp_path):
        """日本語がASCIIエスケープされずに保存されること."""
        output_path = tmp_path / "japanese.json"
        payload = {"label": "犬"}

        write_json_file(output_path, payload)

        raw_text = output_path.read_text(encoding="utf-8")
        assert "犬" in raw_text
        assert "\\u" not in raw_text

    def test_custom_indent(self, tmp_path):
        """カスタムインデント幅が反映されること."""
        output_path = tmp_path / "indented.json"
        payload = {"a": 1}

        write_json_file(output_path, payload, indent=4)

        raw_text = output_path.read_text(encoding="utf-8")
        assert "    " in raw_text
