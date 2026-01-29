"""
Early Stopping モジュール.

検証メトリクスの改善が停止した場合に訓練を自動停止する機能を提供します。
"""

import logging
from typing import Optional


class EarlyStopping:
    """
    Early Stopping クラス.

    検証メトリクスが一定エポック数改善しない場合に訓練停止を通知します。
    PochiTrainerから各エポック後に呼び出され, 停止判定を委譲されます。

    Args:
        patience (int): 改善なしの許容エポック数
        min_delta (float): 改善と見なす最小変化量
        monitor (str): 監視するメトリクス名 ('val_accuracy' or 'val_loss')
        logger (logging.Logger, optional): ロガーインスタンス
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        monitor: str = "val_accuracy",
        logger: Optional[logging.Logger] = None,
    ):
        """EarlyStoppingを初期化."""
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.logger = logger

        # 内部状態
        self.best_value: Optional[float] = None
        self.counter = 0
        self.should_stop = False
        self.best_epoch = 0

        # 監視メトリクスに応じて比較方向を決定
        # val_accuracy: 大きいほど良い, val_loss: 小さいほど良い
        self._is_improvement = (
            self._higher_is_better
            if monitor == "val_accuracy"
            else self._lower_is_better
        )

    def _higher_is_better(self, current: float, best: float) -> bool:
        """現在値がベスト値よりmin_delta以上大きいか判定."""
        return current > best + self.min_delta

    def _lower_is_better(self, current: float, best: float) -> bool:
        """現在値がベスト値よりmin_delta以上小さいか判定."""
        return current < best - self.min_delta

    def step(self, value: float, epoch: int) -> bool:
        """
        エポック終了後にメトリクスを評価し, 停止判定を行う.

        Args:
            value (float): 現在のエポックのメトリクス値
            epoch (int): 現在のエポック番号

        Returns:
            bool: 訓練を停止すべき場合True
        """
        if self.best_value is None:
            # 初回: ベスト値を初期化
            self.best_value = value
            self.best_epoch = epoch
            return False

        if self._is_improvement(value, self.best_value):
            # 改善があった場合: ベスト値を更新しカウンターをリセット
            self.best_value = value
            self.best_epoch = epoch
            self.counter = 0
        else:
            # 改善がなかった場合: カウンターを増加
            self.counter += 1
            if self.logger:
                self.logger.info(
                    f"EarlyStopping: {self.counter}/{self.patience} "
                    f"(ベスト{self.monitor}: {self.best_value:.4f}, "
                    f"エポック {self.best_epoch})"
                )

            if self.counter >= self.patience:
                self.should_stop = True
                if self.logger:
                    self.logger.warning(
                        f"EarlyStopping: {self.patience}エポック間"
                        f"{self.monitor}の改善がないため訓練を停止します. "
                        f"ベスト{self.monitor}: {self.best_value:.4f} "
                        f"(エポック {self.best_epoch})"
                    )
                return True

        return False

    def get_status(self) -> dict:
        """
        現在のEarlyStopping状態を取得.

        Returns:
            dict: 状態情報
        """
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "monitor": self.monitor,
            "counter": self.counter,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "should_stop": self.should_stop,
        }
