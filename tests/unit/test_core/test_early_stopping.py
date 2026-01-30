"""
EarlyStoppingクラスのユニットテスト.
"""

import logging

import pytest

from pochitrain.training.early_stopping import EarlyStopping


class TestEarlyStoppingInit:
    """EarlyStopping初期化のテスト."""

    def test_default_parameters(self):
        """デフォルトパラメータでの初期化テスト."""
        es = EarlyStopping()
        assert es.patience == 10
        assert es.min_delta == 0.0
        assert es.monitor == "val_accuracy"
        assert es.best_value is None
        assert es.counter == 0
        assert es.should_stop is False
        assert es.best_epoch == 0

    def test_custom_parameters(self):
        """カスタムパラメータでの初期化テスト."""
        es = EarlyStopping(patience=5, min_delta=0.01, monitor="val_loss")
        assert es.patience == 5
        assert es.min_delta == 0.01
        assert es.monitor == "val_loss"

    def test_with_logger(self):
        """ロガー付きでの初期化テスト."""
        logger = logging.getLogger("test")
        es = EarlyStopping(logger=logger)
        assert es.logger is logger


class TestEarlyStoppingValAccuracy:
    """val_accuracy監視でのEarlyStoppingテスト."""

    def test_no_stop_on_improvement(self):
        """改善が続く場合は停止しないことを確認."""
        es = EarlyStopping(patience=3, monitor="val_accuracy")
        assert es.step(80.0, 1) is False
        assert es.step(85.0, 2) is False
        assert es.step(90.0, 3) is False
        assert es.should_stop is False
        assert es.counter == 0

    def test_stop_after_patience_exceeded(self):
        """patience回改善なしで停止することを確認."""
        es = EarlyStopping(patience=3, monitor="val_accuracy")
        assert es.step(90.0, 1) is False  # 初期値
        assert es.step(89.0, 2) is False  # 改善なし (1/3)
        assert es.step(88.0, 3) is False  # 改善なし (2/3)
        assert es.step(87.0, 4) is True  # 改善なし (3/3) -> 停止
        assert es.should_stop is True

    def test_counter_resets_on_improvement(self):
        """改善があった場合にカウンターがリセットされることを確認."""
        es = EarlyStopping(patience=3, monitor="val_accuracy")
        assert es.step(90.0, 1) is False
        assert es.step(89.0, 2) is False  # 改善なし (1/3)
        assert es.step(88.0, 3) is False  # 改善なし (2/3)
        assert es.step(91.0, 4) is False  # 改善! -> カウンターリセット
        assert es.counter == 0
        assert es.best_value == 91.0
        assert es.best_epoch == 4

    def test_min_delta_threshold(self):
        """min_deltaによる改善判定のテスト."""
        es = EarlyStopping(patience=2, min_delta=1.0, monitor="val_accuracy")
        assert es.step(90.0, 1) is False  # 初期値
        assert es.step(90.5, 2) is False  # +0.5 < min_delta(1.0) -> 改善なし (1/2)
        assert (
            es.step(90.8, 3) is True
        )  # +0.8 < min_delta(1.0) -> 改善なし (2/2) -> 停止
        assert es.best_value == 90.0  # ベスト値は初期値のまま

    def test_equal_value_counts_as_no_improvement(self):
        """同じ値は改善なしとカウントされることを確認."""
        es = EarlyStopping(patience=2, monitor="val_accuracy")
        assert es.step(90.0, 1) is False
        assert es.step(90.0, 2) is False  # 同値 -> 改善なし (1/2)
        assert es.step(90.0, 3) is True  # 同値 -> 改善なし (2/2) -> 停止


class TestEarlyStoppingValLoss:
    """val_loss監視でのEarlyStoppingテスト."""

    def test_no_stop_on_decreasing_loss(self):
        """損失が減少し続ける場合は停止しないことを確認."""
        es = EarlyStopping(patience=3, monitor="val_loss")
        assert es.step(1.0, 1) is False
        assert es.step(0.8, 2) is False
        assert es.step(0.6, 3) is False
        assert es.should_stop is False

    def test_stop_on_increasing_loss(self):
        """損失が増加し続ける場合に停止することを確認."""
        es = EarlyStopping(patience=2, monitor="val_loss")
        assert es.step(0.5, 1) is False  # 初期値
        assert es.step(0.6, 2) is False  # 悪化 (1/2)
        assert es.step(0.7, 3) is True  # 悪化 (2/2) -> 停止

    def test_min_delta_with_loss(self):
        """val_lossでのmin_delta判定テスト."""
        es = EarlyStopping(patience=2, min_delta=0.1, monitor="val_loss")
        assert es.step(1.0, 1) is False  # 初期値
        assert es.step(0.95, 2) is False  # -0.05 < min_delta(0.1) -> 改善なし (1/2)
        assert (
            es.step(0.92, 3) is True
        )  # -0.08 < min_delta(0.1) -> 改善なし (2/2) -> 停止


class TestEarlyStoppingGetStatus:
    """get_statusメソッドのテスト."""

    def test_initial_status(self):
        """初期状態のステータス確認."""
        es = EarlyStopping(patience=5, min_delta=0.01, monitor="val_accuracy")
        status = es.get_status()
        assert status["patience"] == 5
        assert status["min_delta"] == 0.01
        assert status["monitor"] == "val_accuracy"
        assert status["counter"] == 0
        assert status["best_value"] is None
        assert status["best_epoch"] == 0
        assert status["should_stop"] is False

    def test_status_after_steps(self):
        """数ステップ後のステータス確認."""
        es = EarlyStopping(patience=3, monitor="val_accuracy")
        es.step(90.0, 1)
        es.step(89.0, 2)  # 改善なし

        status = es.get_status()
        assert status["counter"] == 1
        assert status["best_value"] == 90.0
        assert status["best_epoch"] == 1
        assert status["should_stop"] is False


class TestEarlyStoppingWithLogger:
    """ロガー付きEarlyStoppingのテスト."""

    def test_logs_on_no_improvement(self, caplog):
        """改善なし時にログが出力されることを確認."""
        logger = logging.getLogger("test_es")
        logger.setLevel(logging.DEBUG)
        es = EarlyStopping(patience=3, monitor="val_accuracy", logger=logger)

        with caplog.at_level(logging.INFO, logger="test_es"):
            es.step(90.0, 1)
            es.step(89.0, 2)  # 改善なし -> ログ出力

        assert "EarlyStopping: 1/3" in caplog.text

    def test_logs_warning_on_stop(self, caplog):
        """停止時にwarningが出力されることを確認."""
        logger = logging.getLogger("test_es_warn")
        logger.setLevel(logging.DEBUG)
        es = EarlyStopping(patience=1, monitor="val_accuracy", logger=logger)

        with caplog.at_level(logging.WARNING, logger="test_es_warn"):
            es.step(90.0, 1)
            es.step(89.0, 2)  # 停止

        assert "訓練を停止します" in caplog.text
