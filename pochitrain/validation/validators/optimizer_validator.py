"""
Optimizer設定のバリデーター.

learning_rateとoptimizer設定の妥当性をチェックします。
"""

import logging
from typing import Any, Dict

from ..base_validator import BaseValidator


class OptimizerValidator(BaseValidator):
    """Optimizer設定のバリデーションクラス."""

    def __init__(self) -> None:
        """OptimizerValidatorを初期化."""
        # サポートするオプティマイザー名
        self.supported_optimizers = ["Adam", "AdamW", "SGD"]

    def validate(self, config: Dict[str, Any], logger: logging.Logger) -> bool:
        """
        Optimizer設定のバリデーション.

        Args:
            config (Dict[str, Any]): 設定辞書
            logger (logging.Logger): ロガーインスタンス

        Returns:
            bool: バリデーション成功時True、失敗時False
        """
        # learning_rateの必須チェック
        learning_rate = config.get("learning_rate")
        if learning_rate is None:
            logger.error(
                "learning_rate が設定されていません。configs/pochi_train_config.py で "
                "学習率を設定してください。"
            )
            return False

        # learning_rateの型チェック
        if not isinstance(learning_rate, (int, float)):
            logger.error(
                f"learning_rate は数値である必要があります。現在の型: {type(learning_rate)}"
            )
            return False

        # learning_rateの範囲チェック（0 < lr ≤ 1.0）
        if learning_rate <= 0 or learning_rate > 1.0:
            logger.error(
                f"learning_rate は 0 < lr ≤ 1.0 の範囲である必要があります。"
                f"現在の値: {learning_rate}"
            )
            return False

        # optimizerの必須チェック
        optimizer = config.get("optimizer")
        if optimizer is None:
            logger.error(
                "optimizer が設定されていません。configs/pochi_train_config.py で "
                "最適化器を設定してください。"
            )
            return False

        # optimizerの型チェック
        if not isinstance(optimizer, str):
            logger.error(
                f"optimizer は文字列である必要があります。現在の型: {type(optimizer)}"
            )
            return False

        # optimizerの妥当性チェック
        if optimizer not in self.supported_optimizers:
            logger.error(
                f"サポートされていない最適化器です: {optimizer}. "
                f"サポート対象: {self.supported_optimizers}"
            )
            return False

        # 成功時のログ出力
        logger.info(f"学習率: {learning_rate}")
        logger.info(f"最適化器: {optimizer}")

        return True
