"""
データパス設定のバリデーター.

意図しない検証データ不足を防止します。
"""

import logging
from pathlib import Path
from typing import Any, Dict

from ..base_validator import BaseValidator


class DataValidator(BaseValidator):
    """データパス設定のバリデーションクラス."""

    def validate(self, config: Dict[str, Any], logger: logging.Logger) -> bool:
        """
        データパス設定のバリデーション.

        Args:
            config (Dict[str, Any]): 設定辞書
            logger (logging.Logger): ロガーインスタンス

        Returns:
            bool: バリデーション成功時True、失敗時False
        """
        # train_data_root必須チェック
        train_data_root = config.get("train_data_root")
        if not train_data_root:
            logger.error(
                "train_data_root が必須です。configs/pochi_train_config.py で "
                "有効な訓練データパスを設定してください。"
            )
            return False

        # train_data_rootの存在チェック
        train_path = Path(train_data_root)
        if not train_path.exists():
            logger.error(f"訓練データパスが存在しません: {train_data_root}")
            return False

        # val_data_root必須チェック（意図しない検証データ不足を防止）
        val_data_root = config.get("val_data_root")
        if not val_data_root:
            logger.error(
                "val_data_root が必須です。configs/pochi_train_config.py で "
                "有効な検証データパスを設定してください。"
            )
            return False

        # val_data_rootの存在チェック
        val_path = Path(val_data_root)
        if not val_path.exists():
            logger.error(f"検証データパスが存在しません: {val_data_root}")
            return False

        return True
