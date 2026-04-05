"""
pochitrain.logging: ログ管理モジュール.

colorlogを使用したオブジェクト指向のログ管理システム
"""

from .logger_manager import (
    COLORLOG_AVAILABLE,
    LOG_COLORS,
    LOG_DATE_FORMAT,
    LoggerManager,
)

__all__ = ["COLORLOG_AVAILABLE", "LOG_COLORS", "LOG_DATE_FORMAT", "LoggerManager"]
