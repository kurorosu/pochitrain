"""
クラス重み設定のバリデーター.

クラス重み設定の明示化とバリデーション機能を提供します。
"""

import logging
from typing import Any, Dict

from ..base_validator import BaseValidator


class ClassWeightsValidator(BaseValidator):
    """クラス重み設定のバリデーションクラス."""

    def validate(self, config: Dict[str, Any], logger: logging.Logger) -> bool:
        """
        クラス重み設定のバリデーション.

        Args:
            config (Dict[str, Any]): 設定辞書
            logger (logging.Logger): ロガーインスタンス

        Returns:
            bool: バリデーション成功時True、失敗時False
        """
        class_weights = config.get("class_weights")
        num_classes = config.get("num_classes")

        # num_classesの必須チェック
        if num_classes is None:
            logger.error(
                "num_classes が設定されていません。configs/pochi_config.py で "
                "クラス数を設定してください。"
            )
            return False

        # num_classesの型と値チェック
        if not isinstance(num_classes, int) or num_classes <= 0:
            logger.error(
                f"num_classes は正の整数である必要があります。現在の値: {num_classes}"
            )
            return False

        # class_weightsがNoneの場合は正常（クラス重みなし）
        if class_weights is None:
            logger.info("クラス重み: なし（均等扱い）")
            return True

        # class_weightsの型チェック
        if not isinstance(class_weights, (list, tuple)):
            logger.error(
                f"class_weights はリスト形式で設定してください。現在の型: {type(class_weights)}"
            )
            return False

        # class_weightsをリストに変換（tupleの場合）
        weights_list = list(class_weights)

        # 要素数チェック（num_classesとの一致）
        if len(weights_list) != num_classes:
            logger.error(
                f"class_weights の要素数がnum_classesと一致しません。"
                f"class_weights: {len(weights_list)}要素, num_classes: {num_classes}"
            )
            return False

        # 各要素の型と値チェック
        for i, weight in enumerate(weights_list):
            if not isinstance(weight, (int, float)):
                logger.error(
                    f"class_weights[{i}] は数値である必要があります。"
                    f"現在の値: {weight} (型: {type(weight)})"
                )
                return False

            if weight <= 0:
                logger.error(
                    f"class_weights[{i}] は正の値である必要があります。"
                    f"現在の値: {weight}"
                )
                return False

        # バリデーション成功時のログ出力
        logger.info(f"クラス重み: {weights_list}")

        return True
