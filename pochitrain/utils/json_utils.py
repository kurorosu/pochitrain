"""JSONファイル入出力の共通ユーティリティ."""

import json
from pathlib import Path
from typing import Any


def write_json_file(
    output_path: Path,
    payload: Any,
    *,
    ensure_ascii: bool = False,
    indent: int = 2,
) -> Path:
    """JSONファイルを書き出す.

    Args:
        output_path: 出力先ファイルパス.
        payload: JSONとして保存するデータ.
        ensure_ascii: ASCIIエスケープの有無.
        indent: インデント幅.

    Returns:
        保存したファイルパス.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=ensure_ascii, indent=indent)
    return output_path
