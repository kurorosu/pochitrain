"""
Transform設定のバリデーター.

意図しないデフォルトTransform使用を防止します。
"""

import logging
from typing import Any, Dict

from ..base_validator import BaseValidator


class TransformValidator(BaseValidator):
    """Transform設定のバリデーションクラス."""

    def validate(self, config: Dict[str, Any], logger: logging.Logger) -> bool:
        """
        Transform設定のバリデーション.

        Args:
            config (Dict[str, Any]): 設定辞書
            logger (logging.Logger): ロガーインスタンス

        Returns:
            bool: バリデーション成功時True、失敗時False
        """
        # train_transform必須チェック（意図しないデフォルト使用を防止）
        train_transform = config.get("train_transform")
        if train_transform is None:
            logger.error(
                "train_transform が必須です。configs/pochi_train_config.py で "
                "transforms.Compose([...]) を train_transform として定義してください。"
            )
            return False

        # val_transform必須チェック（意図しないデフォルト使用を防止）
        val_transform = config.get("val_transform")
        if val_transform is None:
            logger.error(
                "val_transform が必須です。configs/pochi_train_config.py で "
                "transforms.Compose([...]) を val_transform として定義してください。"
            )
            return False

        return True
