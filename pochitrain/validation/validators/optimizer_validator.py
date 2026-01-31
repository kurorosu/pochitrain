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
        # learning_rateのバリデーション
        learning_rate = config.get("learning_rate")
        if not self._validate_required_type(
            learning_rate, "learning_rate", (int, float), logger
        ):
            return False
        if not self._validate_range(
            learning_rate, "learning_rate", logger, gt=0, le=1.0
        ):
            return False

        # optimizerのバリデーション
        optimizer = config.get("optimizer")
        if not self._validate_required_type(optimizer, "optimizer", str, logger):
            return False
        if not self._validate_choice(
            optimizer, "最適化器", self.supported_optimizers, logger
        ):
            return False

        # 成功時のログ出力
        logger.info(f"学習率: {learning_rate}")
        logger.info(f"最適化器: {optimizer}")

        return True
