"""
Early Stopping設定のバリデーター.

early_stopping設定の妥当性をチェックします。
"""

import logging
from typing import Any, Dict

from ..base_validator import BaseValidator


class EarlyStoppingValidator(BaseValidator):
    """Early Stopping設定のバリデーションクラス."""

    SUPPORTED_MONITORS = ["val_accuracy", "val_loss"]

    def validate(self, config: Dict[str, Any], logger: logging.Logger) -> bool:
        """
        Early Stopping設定のバリデーション.

        early_stopping設定が存在しない場合や無効の場合はスキップして成功を返します。

        Args:
            config (Dict[str, Any]): 設定辞書
            logger (logging.Logger): ロガーインスタンス

        Returns:
            bool: バリデーション成功時True、失敗時False
        """
        es_config = config.get("early_stopping")

        # 設定がない場合はスキップ（後方互換性）
        if es_config is None:
            return True

        # 辞書型チェック
        if not isinstance(es_config, dict):
            logger.error(
                f"early_stopping は辞書である必要があります。"
                f"現在の型: {type(es_config).__name__}"
            )
            return False

        # enabled チェック
        enabled = es_config.get("enabled", False)
        if not isinstance(enabled, bool):
            logger.error(
                f"early_stopping.enabled はboolである必要があります。"
                f"現在の型: {type(enabled).__name__}, 現在の値: {enabled}"
            )
            return False

        # 無効の場合はこれ以上チェック不要
        if not enabled:
            return True

        # patience チェック
        patience = es_config.get("patience", 10)
        if not self._validate_required_type(
            patience, "early_stopping.patience", int, logger, exclude_bool=True
        ):
            return False
        if not self._validate_positive(patience, "early_stopping.patience", logger):
            return False

        # min_delta チェック
        min_delta = es_config.get("min_delta", 0.0)
        if not isinstance(min_delta, (int, float)) or isinstance(min_delta, bool):
            logger.error(
                f"early_stopping.min_delta は数値である必要があります。"
                f"現在の型: {type(min_delta).__name__}, 現在の値: {min_delta}"
            )
            return False
        if not self._validate_range(
            min_delta, "early_stopping.min_delta", logger, ge=0
        ):
            return False

        # monitor チェック
        monitor = es_config.get("monitor", "val_accuracy")
        if not self._validate_required_type(
            monitor, "early_stopping.monitor", str, logger
        ):
            return False
        if not self._validate_choice(
            monitor, "monitor値", self.SUPPORTED_MONITORS, logger
        ):
            return False

        logger.info(
            f"Early Stopping: 有効 "
            f"(patience={patience}, "
            f"min_delta={min_delta}, "
            f"monitor={monitor})"
        )

        return True
