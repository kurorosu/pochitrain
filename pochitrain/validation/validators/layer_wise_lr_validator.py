"""
層別学習率設定のバリデーター.

層別学習率の設定が正しいかチェックします。
"""

import logging
from typing import Any, Dict

from ..base_validator import BaseValidator


class LayerWiseLRValidator(BaseValidator):
    """層別学習率設定のバリデーションクラス."""

    def validate(self, config: Dict[str, Any], logger: logging.Logger) -> bool:
        """
        層別学習率設定のバリデーション.

        Args:
            config (Dict[str, Any]): 設定辞書
            logger (logging.Logger): ロガーインスタンス

        Returns:
            bool: バリデーションが成功した場合True、失敗した場合False
        """
        # enable_layer_wise_lr の存在と型チェック
        if "enable_layer_wise_lr" not in config:
            logger.error("設定に 'enable_layer_wise_lr' が見つかりません")
            return False

        enable_layer_wise_lr = config["enable_layer_wise_lr"]
        if not isinstance(enable_layer_wise_lr, bool):
            logger.error(
                f"'enable_layer_wise_lr' はbool型である必要があります。"
                f"現在の型: {type(enable_layer_wise_lr)}"
            )
            return False

        # 層別学習率が無効の場合は、これ以上のチェックは不要
        if not enable_layer_wise_lr:
            logger.debug("層別学習率は無効です。バリデーションをスキップします。")
            return True

        # layer_wise_lr_config の存在チェック
        if "layer_wise_lr_config" not in config:
            logger.error(
                "enable_layer_wise_lr=True の場合、'layer_wise_lr_config' が必要です"
            )
            return False

        layer_wise_lr_config = config["layer_wise_lr_config"]
        if not isinstance(layer_wise_lr_config, dict):
            logger.error(
                f"'layer_wise_lr_config' は辞書型である必要があります。"
                f"現在の型: {type(layer_wise_lr_config)}"
            )
            return False

        # layer_rates の存在と型チェック
        if "layer_rates" not in layer_wise_lr_config:
            logger.error("'layer_wise_lr_config' に 'layer_rates' が見つかりません")
            return False

        layer_rates = layer_wise_lr_config["layer_rates"]
        if not isinstance(layer_rates, dict):
            logger.error(
                f"'layer_rates' は辞書型である必要があります。現在の型: {type(layer_rates)}"
            )
            return False

        # layer_rates の各エントリをチェック
        if not layer_rates:
            logger.error(
                "'layer_rates' が空です。少なくとも1つの層の学習率を指定してください"
            )
            return False

        for layer_name, learning_rate in layer_rates.items():
            if not isinstance(layer_name, str):
                logger.error(
                    f"層名は文字列である必要があります。層名: {layer_name}, 型: {type(layer_name)}"
                )
                return False

            if not isinstance(learning_rate, (int, float)):
                logger.error(
                    f"学習率は数値である必要があります。層: {layer_name}, "
                    f"学習率: {learning_rate}, 型: {type(learning_rate)}"
                )
                return False

            if learning_rate <= 0:
                logger.error(
                    f"学習率は正の値である必要があります。層: {layer_name}, 学習率: {learning_rate}"
                )
                return False

        # 推奨される層名のチェック（警告レベル）
        recommended_layers = {
            "conv1",
            "bn1",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
            "fc",
        }
        configured_layers = set(layer_rates.keys())

        missing_layers = recommended_layers - configured_layers
        if missing_layers:
            logger.warning(
                f"推奨される層が設定されていません: {sorted(missing_layers)}. "
                "これらの層は基本学習率が適用されます。"
            )

        unknown_layers = configured_layers - recommended_layers
        if unknown_layers:
            logger.warning(
                f"未知の層名が設定されています: {sorted(unknown_layers)}. "
                "これらの設定は無視される可能性があります。"
            )

        logger.debug(f"層別学習率設定が有効です。設定された層数: {len(layer_rates)}")
        return True
