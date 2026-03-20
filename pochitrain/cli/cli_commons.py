"""CLI 共通ユーティリティ.

全サブコマンドで共有されるロギング設定やシグナルハンドラーを提供する.
"""

import logging
from types import FrameType
from typing import Any, Optional

from pochitrain.logging import LoggerManager
from pochitrain.logging.logger_manager import LogLevel

training_interrupted = False


def create_signal_handler(debug: bool = False) -> Any:
    """デバッグフラグを保持するシグナルハンドラーを生成する.

    Args:
        debug (bool): デバッグモードが有効かどうか

    Returns:
        シグナルハンドラー関数
    """

    def signal_handler(signum: int, frame: Optional[FrameType]) -> None:
        """Ctrl+Cのシグナルハンドラー."""
        global training_interrupted
        training_interrupted = True

        logger = setup_logging(debug=debug)
        logger.warning("訓練を安全に停止しています... (Ctrl+Cが検出されました)")
        logger.warning("現在のエポックが完了次第、訓練を終了します。")

    return signal_handler


def setup_logging(
    logger_name: str = "pochitrain", debug: bool = False
) -> logging.Logger:
    """ログ設定の初期化.

    Args:
        logger_name (str): ロガー名
        debug (bool): デバッグモードが有効かどうか

    Returns:
        logger: 設定済みロガー
    """
    logger_manager = LoggerManager()
    level = LogLevel.DEBUG if debug else LogLevel.INFO
    logger_manager.set_default_level(level)
    for existing_name in logger_manager.get_available_loggers():
        logger_manager.set_logger_level(existing_name, level)
    return logger_manager.get_logger(logger_name, level=level)
