"""
LoggerManagerのテスト
"""

import logging

from pochitrain.logging import LoggerManager
from pochitrain.logging.logger_manager import LogLevel


class TestLoggerManager:
    """LoggerManagerクラスのテスト"""

    def setup_method(self):
        """各テストメソッドの前に実行される初期化処理"""
        # シングルトンインスタンスをリセット
        LoggerManager.reset()

    def test_singleton_pattern(self):
        """シングルトンパターンの動作テスト"""
        manager1 = LoggerManager()
        manager2 = LoggerManager()

        # 同じインスタンスであることを確認
        assert manager1 is manager2

    def test_get_logger_basic(self):
        """基本的なロガー取得機能のテスト"""
        manager = LoggerManager()
        logger = manager.get_logger("test_logger")

        # ロガーが正しく作成されることを確認
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"
        assert logger.level == logging.INFO  # デフォルトレベル

    def test_get_logger_with_custom_level(self):
        """カスタムレベルでのロガー取得テスト"""
        manager = LoggerManager()
        logger = manager.get_logger("debug_logger", LogLevel.DEBUG)

        assert logger.level == logging.DEBUG

    def test_logger_reuse(self):
        """同じ名前のロガーが再利用されることをテスト"""
        manager = LoggerManager()
        logger1 = manager.get_logger("reuse_test")
        logger2 = manager.get_logger("reuse_test")

        # 同じロガーインスタンスが返されることを確認
        assert logger1 is logger2

    def test_multiple_loggers(self):
        """複数のロガーが管理されることをテスト"""
        manager = LoggerManager()
        logger1 = manager.get_logger("logger1")
        logger2 = manager.get_logger("logger2")

        # 異なるロガーインスタンスが作成されることを確認
        assert logger1 is not logger2
        assert logger1.name == "logger1"
        assert logger2.name == "logger2"

        # 管理されているロガーリストに含まれることを確認
        available_loggers = manager.get_available_loggers()
        assert "logger1" in available_loggers
        assert "logger2" in available_loggers

    def test_set_logger_level(self):
        """ロガーレベルの動的変更テスト"""
        manager = LoggerManager()
        logger = manager.get_logger("level_test")

        # 初期レベルの確認
        assert logger.level == logging.INFO

        # レベルの変更
        manager.set_logger_level("level_test", LogLevel.ERROR)
        assert logger.level == logging.ERROR

    def test_set_default_level(self):
        """デフォルトレベルの変更テスト"""
        manager = LoggerManager()

        # デフォルトレベルを変更
        manager.set_default_level(LogLevel.WARNING)

        # 新しいロガーが変更されたデフォルトレベルを使用することを確認
        logger = manager.get_logger("default_level_test")
        assert logger.level == logging.WARNING

    def test_fallback_when_colorlog_unavailable(self, monkeypatch):
        """colorlogが利用できない場合のフォールバック機能テスト"""
        # colorlogが利用できない状態をモック
        monkeypatch.setattr(
            "pochitrain.logging.logger_manager.COLORLOG_AVAILABLE",
            False,
        )

        # 新しいマネージャーを作成（colorlogが無効な状態で）
        LoggerManager.reset()
        manager = LoggerManager()
        logger = manager.get_logger("fallback_test")

        # ログハンドラーが正常に作成されることを確認
        assert len(logger.handlers) > 0
        assert isinstance(logger.handlers[0], logging.StreamHandler)

    def test_colorlog_when_available(self, monkeypatch):
        """colorlogが利用可能な場合のテスト"""
        # colorlogが利用可能な状態をモック
        monkeypatch.setattr(
            "pochitrain.logging.logger_manager.COLORLOG_AVAILABLE",
            True,
        )

        LoggerManager.reset()
        manager = LoggerManager()

        # colorlog利用可能フラグの確認
        assert manager.is_colorlog_available() is True

    def test_reset_functionality(self):
        """resetメソッドの動作テスト"""
        manager1 = LoggerManager()
        manager1.get_logger("reset_test")

        # ロガーが管理されていることを確認
        assert "reset_test" in manager1.get_available_loggers()

        # リセット実行
        LoggerManager.reset()

        # 新しいマネージャーインスタンスが作成されることを確認
        manager2 = LoggerManager()
        assert manager1 is not manager2
        assert len(manager2.get_available_loggers()) == 0

    def test_log_output(self, caplog):
        """実際のログ出力のテスト"""
        manager = LoggerManager()
        logger = manager.get_logger("output_test")

        # ログレベルをDEBUGに設定
        manager.set_logger_level("output_test", LogLevel.DEBUG)

        # propagateをTrueにしてcaplogでキャプチャできるようにする
        logger.propagate = True

        with caplog.at_level(logging.DEBUG, logger="output_test"):
            logger.debug("デバッグメッセージ")
            logger.info("情報メッセージ")
            logger.warning("警告メッセージ")
            logger.error("エラーメッセージ")

        # ログが記録されていることを確認（少なくとも一部が記録される）
        assert len(caplog.records) >= 2  # WARNING以上は確実に記録される

        # メッセージが含まれていることを確認
        log_messages = [record.message for record in caplog.records]
        assert any("警告メッセージ" in msg for msg in log_messages)
        assert any("エラーメッセージ" in msg for msg in log_messages)
