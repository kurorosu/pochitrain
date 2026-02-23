"""
LoggerManagerのテスト
"""

import logging

from pochitrain.logging import LoggerManager
from pochitrain.logging.logger_manager import LogLevel


class TestLoggerManager:
    """LoggerManagerクラスのテスト"""

    def test_singleton_pattern(self):
        """シングルトンパターンの動作テスト"""
        manager1 = LoggerManager()
        manager2 = LoggerManager()

        assert manager1 is manager2

    def test_get_logger_basic(self):
        """基本的なロガー取得機能のテスト"""
        manager = LoggerManager()
        manager.set_default_level(LogLevel.INFO)
        logger = manager.get_logger("test_logger_basic")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger_basic"
        assert logger.level == logging.INFO  # デフォルトレベル

    def test_get_logger_with_custom_level(self):
        """カスタムレベルでのロガー取得テスト"""
        manager = LoggerManager()
        logger = manager.get_logger("debug_logger_custom", LogLevel.DEBUG)

        assert logger.level == logging.DEBUG

    def test_logger_reuse(self):
        """同じ名前のロガーが再利用されることをテスト"""
        manager = LoggerManager()
        logger1 = manager.get_logger("reuse_test_logger")
        logger2 = manager.get_logger("reuse_test_logger")

        assert logger1 is logger2

    def test_multiple_loggers(self):
        """複数のロガーが管理されることをテスト"""
        manager = LoggerManager()
        logger1 = manager.get_logger("logger_multi_1")
        logger2 = manager.get_logger("logger_multi_2")

        assert logger1 is not logger2
        assert logger1.name == "logger_multi_1"
        assert logger2.name == "logger_multi_2"

        available_loggers = manager.get_available_loggers()
        assert "logger_multi_1" in available_loggers
        assert "logger_multi_2" in available_loggers

    def test_set_logger_level(self):
        """ロガーレベルの動的変更テスト"""
        manager = LoggerManager()
        manager.set_default_level(LogLevel.INFO)
        logger = manager.get_logger("level_test_logger")

        assert logger.level == logging.INFO

        manager.set_logger_level("level_test_logger", LogLevel.ERROR)
        assert logger.level == logging.ERROR

    def test_set_default_level(self):
        """デフォルトレベルの変更テスト"""
        manager = LoggerManager()

        manager.set_default_level(LogLevel.WARNING)

        logger = manager.get_logger("default_level_logger")
        assert logger.level == logging.WARNING

    def test_fallback_when_colorlog_unavailable(self, monkeypatch):
        """colorlogが利用できない場合のフォールバック機能テスト"""
        monkeypatch.setattr(
            "pochitrain.logging.logger_manager.COLORLOG_AVAILABLE",
            False,
        )

        manager = LoggerManager()
        logger = manager.get_logger("fallback_test_logger")

        assert len(logger.handlers) > 0
        assert isinstance(logger.handlers[0], logging.StreamHandler)

    def test_colorlog_when_available(self, monkeypatch):
        """colorlogが利用可能な場合のテスト"""
        monkeypatch.setattr(
            "pochitrain.logging.logger_manager.COLORLOG_AVAILABLE",
            True,
        )

        manager = LoggerManager()

        assert manager.is_colorlog_available() is True

    def test_log_output(self, caplog):
        """実際のログ出力のテスト"""
        manager = LoggerManager()
        logger = manager.get_logger("output_test_logger")

        manager.set_logger_level("output_test_logger", LogLevel.DEBUG)

        logger.propagate = True

        with caplog.at_level(logging.DEBUG, logger="output_test_logger"):
            logger.debug("デバッグメッセージ")
            logger.info("情報メッセージ")
            logger.warning("警告メッセージ")
            logger.error("エラーメッセージ")

        assert len(caplog.records) >= 2  # WARNING以上は確実に記録される

        log_messages = [record.message for record in caplog.records]
        assert any("警告メッセージ" in msg for msg in log_messages)
        assert any("エラーメッセージ" in msg for msg in log_messages)
