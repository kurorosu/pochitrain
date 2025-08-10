"""
デバイス設定のバリデーター.

GPU/CPU設定の意図しない動作を防止します。
"""

import logging
from typing import Any, Dict

from ..base_validator import BaseValidator


class DeviceValidator(BaseValidator):
    """デバイス設定のバリデーションクラス."""

    def validate(self, config: Dict[str, Any], logger: logging.Logger) -> bool:
        """
        デバイス設定のバリデーション.

        Args:
            config (Dict[str, Any]): 設定辞書
            logger (logging.Logger): ロガーインスタンス

        Returns:
            bool: バリデーション成功時True、失敗時False
        """
        device_config = config.get("device")

        # device設定がNoneの場合はエラー（意図しないCPU使用を防止）
        if device_config is None:
            logger.error(
                "device設定が必須です。configs/pochi_config.pyでdeviceを'cuda'または'cpu'に設定してください。"
            )
            logger.error("例: device = 'cuda' または device = 'cpu'")
            return False

        # CPU使用時の明示的な警告（意図しないパフォーマンス低下を防止）
        if device_config == "cpu":
            logger.warning("⚠️  CPU使用モードで実行中です")
            logger.warning("⚠️  GPU使用を推奨します（大幅な性能向上が期待できます）")
            logger.warning("⚠️  GPU使用時: device = 'cuda' に設定してください")

        return True
