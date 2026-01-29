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
        if not self._validate_patience(es_config, logger):
            return False

        # min_delta チェック
        if not self._validate_min_delta(es_config, logger):
            return False

        # monitor チェック
        if not self._validate_monitor(es_config, logger):
            return False

        logger.info(
            f"Early Stopping: 有効 "
            f"(patience={es_config.get('patience', 10)}, "
            f"min_delta={es_config.get('min_delta', 0.0)}, "
            f"monitor={es_config.get('monitor', 'val_accuracy')})"
        )

        return True

    def _validate_patience(
        self, es_config: Dict[str, Any], logger: logging.Logger
    ) -> bool:
        """
        patienceパラメータのバリデーション.

        Args:
            es_config (Dict[str, Any]): early_stopping設定辞書
            logger (logging.Logger): ロガーインスタンス

        Returns:
            bool: バリデーション成功時True、失敗時False
        """
        patience = es_config.get("patience", 10)

        if not isinstance(patience, int) or isinstance(patience, bool):
            logger.error(
                f"early_stopping.patience は整数である必要があります。"
                f"現在の型: {type(patience).__name__}, 現在の値: {patience}"
            )
            return False

        if patience <= 0:
            logger.error(
                f"early_stopping.patience は正の整数である必要があります。"
                f"現在の値: {patience}"
            )
            return False

        return True

    def _validate_min_delta(
        self, es_config: Dict[str, Any], logger: logging.Logger
    ) -> bool:
        """
        min_deltaパラメータのバリデーション.

        Args:
            es_config (Dict[str, Any]): early_stopping設定辞書
            logger (logging.Logger): ロガーインスタンス

        Returns:
            bool: バリデーション成功時True、失敗時False
        """
        min_delta = es_config.get("min_delta", 0.0)

        if not isinstance(min_delta, (int, float)) or isinstance(min_delta, bool):
            logger.error(
                f"early_stopping.min_delta は数値である必要があります。"
                f"現在の型: {type(min_delta).__name__}, 現在の値: {min_delta}"
            )
            return False

        if min_delta < 0:
            logger.error(
                f"early_stopping.min_delta は0以上である必要があります。"
                f"現在の値: {min_delta}"
            )
            return False

        return True

    def _validate_monitor(
        self, es_config: Dict[str, Any], logger: logging.Logger
    ) -> bool:
        """
        monitorパラメータのバリデーション.

        Args:
            es_config (Dict[str, Any]): early_stopping設定辞書
            logger (logging.Logger): ロガーインスタンス

        Returns:
            bool: バリデーション成功時True、失敗時False
        """
        monitor = es_config.get("monitor", "val_accuracy")

        if not isinstance(monitor, str):
            logger.error(
                f"early_stopping.monitor は文字列である必要があります。"
                f"現在の型: {type(monitor).__name__}, 現在の値: {monitor}"
            )
            return False

        if monitor not in self.SUPPORTED_MONITORS:
            logger.error(
                f"サポートされていないmonitor値です: {monitor}. "
                f"サポート対象: {self.SUPPORTED_MONITORS}"
            )
            return False

        return True
