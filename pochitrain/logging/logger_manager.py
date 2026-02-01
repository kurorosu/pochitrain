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


class LevelBasedFormatter(logging.Formatter):
    """デバッグモードによって切り替わるログ形式."""

    def __init__(
        self,
        info_format: str,
        debug_format: str,
        datefmt: str,
        use_color: bool = False,
        log_colors: dict | None = None,
        force_debug_format: bool = False,
    ) -> None:
        """ログ整形の初期化."""
        super().__init__(datefmt=datefmt)
        self._info_format = info_format
        self._debug_format = debug_format
        self._use_color = use_color
        self._log_colors = log_colors or {}
        self._force_debug_format = force_debug_format
        if use_color:
            self._info_formatter = colorlog.ColoredFormatter(
                info_format, datefmt=datefmt, log_colors=self._log_colors
            )
            self._debug_formatter = colorlog.ColoredFormatter(
                debug_format, datefmt=datefmt, log_colors=self._log_colors
            )
        else:
            self._info_formatter = logging.Formatter(info_format, datefmt=datefmt)
            self._debug_formatter = logging.Formatter(debug_format, datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        """ログレコードを整形."""
        record.levelname = {"WARNING": "WARN"}.get(record.levelname, record.levelname)
        if self._force_debug_format:
            return str(self._debug_formatter.format(record))
        return str(self._info_formatter.format(record))


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
        self._use_debug_format = False
        self._info_format = (
            "%(asctime)s|%(log_color)s%(levelname)-5.5s%(reset)s| %(message)s"
        )
        self._debug_format = (
            "%(asctime)s|%(log_color)s%(levelname)-5.5s%(reset)s|"
            "%(filename)-28s|%(lineno)03d| %(message)s"
        )
        self._date_format = "%Y-%m-%d %H:%M:%S"
        self._log_colors = {
            "DEBUG": "cyan",
            "INFO": "green",
            "WARN": "yellow",
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
            formatter = LevelBasedFormatter(
                self._info_format,
                self._debug_format,
                datefmt=self._date_format,
                use_color=True,
                log_colors=self._log_colors,
                force_debug_format=self._use_debug_format,
            )
        else:
            handler = logging.StreamHandler()
            # colorlogが利用できない場合は色情報を除去したフォーマット
            info_format = "%(asctime)s|%(levelname)-5.5s| %(message)s"
            debug_format = "%(asctime)s|%(levelname)-5.5s|%(filename)-28s|%(lineno)03d| %(message)s"
            formatter = LevelBasedFormatter(
                info_format,
                debug_format,
                datefmt=self._date_format,
                use_color=False,
                force_debug_format=self._use_debug_format,
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
        self._use_debug_format = level == LogLevel.DEBUG
        self._update_existing_handlers_format()

    def _update_existing_handlers_format(self) -> None:
        """既存ハンドラーのフォーマット設定を更新する."""
        for logger in self._loggers.values():
            for handler in logger.handlers:
                formatter = handler.formatter
                if isinstance(formatter, LevelBasedFormatter):
                    formatter._force_debug_format = self._use_debug_format

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
