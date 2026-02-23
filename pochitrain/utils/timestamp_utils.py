"""
pochitrain.utils.timestamp_utils: タイムスタンプ関連ユーティリティ.

yyyymmdd{index} 形式のディレクトリ名生成と管理機能
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Tuple


def find_next_index(base_dir: Path, date_str: str) -> int:
    """
    同じ日付で既存のディレクトリがある場合、次のインデックスを取得.

    Args:
        base_dir (Path): 検索対象のベースディレクトリ
        date_str (str): 日付文字列 (例: "20241220")

    Returns:
        int: 次のインデックス (1, 2, 3, ...)

    Examples:
        work_dirs/20241220_001/ が存在 → 2 を返す
        work_dirs/20241220_002/ も存在 → 3 を返す
        該当なし → 1 を返す
    """
    if not base_dir.exists():
        return 1

    pattern = re.compile(rf"^{date_str}_(\d{{3}})$")
    indices = []

    for item in base_dir.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                indices.append(int(match.group(1)))

    if not indices:
        return 1

    return max(indices) + 1


def parse_timestamp_dir(dirname: str) -> Tuple[str, int]:
    """
    ディレクトリ名から日付とインデックスを分離.

    Args:
        dirname (str): ディレクトリ名 (例: "20241220_001")

    Returns:
        Tuple[str, int]: (日付文字列, インデックス) のタプル

    Raises:
        ValueError: 形式が正しくない場合

    Examples:
        "20241220_001" → ("20241220", 1)
        "20241220_123" → ("20241220", 123)
    """
    pattern = re.compile(r"^(\d{8})_(\d{3})$")
    match = pattern.match(dirname)

    if not match:
        raise ValueError(f"Invalid directory name format: {dirname}")

    date_str = match.group(1)
    index = int(match.group(2))

    return date_str, index


def get_current_date_str() -> str:
    """
    現在の日付を yyyymmdd 形式で取得.

    Returns:
        str: 現在の日付文字列 (例: "20241220")
    """
    return datetime.now().strftime("%Y%m%d")


def get_current_timestamp() -> str:
    """
    現在の日時を yyyymmdd_hhmmss 形式で取得.

    Returns:
        str: 現在の日時文字列 (例: "20241220_153045")
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_workspace_name(date_str: str, index: int) -> str:
    """
    日付とインデックスからワークスペース名を生成.

    Args:
        date_str (str): 日付文字列 (例: "20241220")
        index (int): インデックス

    Returns:
        str: ワークスペース名 (例: "20241220_001")
    """
    return f"{date_str}_{index:03d}"
