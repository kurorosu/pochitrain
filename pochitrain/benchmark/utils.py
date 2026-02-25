"""ベンチマーク実行の共通ユーティリティ."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from pochitrain.logging import LoggerManager
from pochitrain.logging.logger_manager import LogLevel

LOGGER_NAME = "pochitrain.benchmark"
JST = timezone(timedelta(hours=9))


def configure_logger(debug: bool = False) -> logging.Logger:
    """ベンチマーク用ロガーを初期化して返す.

    Args:
        debug: デバッグログを有効化するかどうか.

    Returns:
        構成済みロガー.
    """
    manager = LoggerManager()
    level = LogLevel.DEBUG if debug else LogLevel.INFO
    manager.set_default_level(level)
    manager.set_logger_level(LOGGER_NAME, level)
    return manager.get_logger(LOGGER_NAME, level=level)


def now_jst_timestamp() -> str:
    """JST現在時刻を `YYYY-MM-DD HH:MM:SS` 形式で返す.

    Returns:
        秒精度の JST 時刻文字列.
    """
    return datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")


def now_local_timestamp() -> str:
    """フォルダ名に使うJSTタイムスタンプを返す.

    Returns:
        `YYYYMMDD_HHMMSS` 形式の時刻文字列.
    """
    return datetime.now(JST).strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """JSON を UTF-8 で保存する.

    Args:
        path: 出力ファイルパス.
        payload: 保存データ.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def to_float(value: Any) -> Optional[float]:
    """任意値を float に変換する.

    Args:
        value: 変換対象値.

    Returns:
        変換後の float 値. 変換不能時は None.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
