"""
Training設定のバリデーター.

epochs、batch_size、model_name設定の妥当性をチェックします。
"""

import logging
from typing import Any, Dict

from ..base_validator import BaseValidator


class TrainingValidator(BaseValidator):
    """Training設定のバリデーションクラス."""

    def __init__(self) -> None:
        """TrainingValidatorを初期化."""
        # サポートするモデル名（pochitrain.models.pochi_modelsと同期）
        self.supported_models = ["resnet18", "resnet34", "resnet50"]

    def validate(self, config: Dict[str, Any], logger: logging.Logger) -> bool:
        """
        Training設定のバリデーション.

        Args:
            config (Dict[str, Any]): 設定辞書
            logger (logging.Logger): ロガーインスタンス

        Returns:
            bool: バリデーション成功時True、失敗時False
        """
        # epochsのバリデーション
        epochs = config.get("epochs")
        if not self._validate_required_type(
            epochs, "epochs", int, logger, exclude_bool=True
        ):
            return False
        if not self._validate_positive(epochs, "epochs", logger):
            return False

        # batch_sizeのバリデーション
        batch_size = config.get("batch_size")
        if not self._validate_required_type(
            batch_size, "batch_size", int, logger, exclude_bool=True
        ):
            return False
        if not self._validate_positive(batch_size, "batch_size", logger):
            return False

        # model_nameのバリデーション
        model_name = config.get("model_name")
        if not self._validate_required_type(model_name, "model_name", str, logger):
            return False
        if not self._validate_choice(
            model_name, "モデル名", self.supported_models, logger
        ):
            return False

        # 成功時のログ出力
        logger.info(f"エポック数: {epochs}")
        logger.info(f"バッチサイズ: {batch_size}")
        logger.info(f"モデル名: {model_name}")

        return True
