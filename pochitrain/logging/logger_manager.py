"""
pochitrain.logging.logger_manager: ログ管理マネージャー.

colorlogを使用したオブジェクト指向のログ管理システム
"""

import logging
from enum import Enum
from typing import Dict, Optional

try:
    import colorlog

    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False


class LogLevel(Enum):
    """ログレベル列挙型."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LoggerManager:
    """
    ログ管理マネージャークラス.

    colorlogを使用したカラフルなログ出力を管理し、
    アプリケーション全体で一貫したログ設定を提供

    Attributes:
        _loggers (Dict[str, logging.Logger]): 管理されているロガーの辞書
        _default_level (LogLevel): デフォルトのログレベル
        _format_string (str): ログフォーマット文字列
    """

    _instance: Optional["LoggerManager"] = None
    _loggers: Dict[str, logging.Logger] = {}

    def __new__(cls) -> "LoggerManager":
        """シングルトンパターンの実装."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """LoggerManagerを初期化."""
        if hasattr(self, "_initialized"):
            return

        self._default_level = LogLevel.INFO
        self._format_string = (
            "[%(asctime)s][%(log_color)s%(levelname)s%(reset)s]"
            "[%(name)s][%(filename)s:%(lineno)d] %(message)s"
        )
        self._date_format = "%Y-%m-%d %H:%M:%S"
        self._log_colors = {
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        }
        self._initialized = True

    def get_logger(self, name: str, level: Optional[LogLevel] = None) -> logging.Logger:
        """
        指定された名前のロガーを取得または作成.

        Args:
            name (str): ロガー名
            level (LogLevel, optional): ログレベル

        Returns:
            logging.Logger: 設定されたロガー

        Examples:
            >>> manager = LoggerManager()
            >>> logger = manager.get_logger("pochitrain")
            >>> logger.info("ログメッセージ")
            [2025-07-14 18:37:48,735][INFO][pochitrain][main.py:123] ログメッセージ
        """
        if name in self._loggers:
            return self._loggers[name]

        logger = self._create_logger(name, level or self._default_level)
        self._loggers[name] = logger
        return logger

    def _create_logger(self, name: str, level: LogLevel) -> logging.Logger:
        """
        新しいロガーを作成.

        Args:
            name (str): ロガー名
            level (LogLevel): ログレベル

        Returns:
            logging.Logger: 作成されたロガー
        """
        logger = logging.getLogger(name)

        # 既にハンドラーが設定されている場合はそのまま返す
        if logger.handlers:
            return logger

        # ログレベルの設定
        log_level = getattr(logging, level.value)
        logger.setLevel(log_level)

        # ハンドラーとフォーマッターの作成
        handler = self._create_handler()
        logger.addHandler(handler)

        # 親ロガーへの伝播を防ぐ
        logger.propagate = False

        return logger

    def _create_handler(self) -> logging.Handler:
        """
        ログハンドラーを作成.

        Returns:
            logging.Handler: 作成されたハンドラー
        """
        handler: logging.Handler
        formatter: logging.Formatter
        if COLORLOG_AVAILABLE:
            handler = colorlog.StreamHandler()
            formatter = colorlog.ColoredFormatter(
                self._format_string,
                datefmt=self._date_format,
                log_colors=self._log_colors,
            )
        else:
            handler = logging.StreamHandler()
            # colorlogが利用できない場合は色情報を除去したフォーマット
            plain_format = (
                "[%(asctime)s][%(levelname)s][%(name)s]"
                "[%(filename)s:%(lineno)d] %(message)s"
            )
            formatter = logging.Formatter(
                plain_format,
                datefmt=self._date_format,
            )

        handler.setFormatter(formatter)
        return handler

    def set_default_level(self, level: LogLevel) -> None:
        """
        デフォルトのログレベルを設定.

        Args:
            level (LogLevel): 新しいデフォルトレベル
        """
        self._default_level = level

    def set_logger_level(self, name: str, level: LogLevel) -> None:
        """
        特定のロガーのレベルを設定.

        Args:
            name (str): ロガー名
            level (LogLevel): 新しいログレベル
        """
        if name in self._loggers:
            log_level = getattr(logging, level.value)
            self._loggers[name].setLevel(log_level)

    def get_available_loggers(self) -> list[str]:
        """
        管理されているロガーの名前一覧を取得.

        Returns:
            list[str]: ロガー名のリスト
        """
        return list(self._loggers.keys())

    def is_colorlog_available(self) -> bool:
        """
        colorlogが利用可能かチェック.

        Returns:
            bool: colorlogが利用可能な場合True
        """
        return COLORLOG_AVAILABLE

    @classmethod
    def reset(cls) -> None:
        """シングルトンインスタンスをリセット（主にテスト用）."""
        cls._instance = None
        cls._loggers.clear()
