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
        if not self._validate_epochs(config, logger):
            return False

        # batch_sizeのバリデーション
        if not self._validate_batch_size(config, logger):
            return False

        # model_nameのバリデーション
        if not self._validate_model_name(config, logger):
            return False

        # 成功時のログ出力
        logger.info(f"エポック数: {config.get('epochs')}")
        logger.info(f"バッチサイズ: {config.get('batch_size')}")
        logger.info(f"モデル名: {config.get('model_name')}")

        return True

    def _validate_epochs(self, config: Dict[str, Any], logger: logging.Logger) -> bool:
        """
        epochsパラメータのバリデーション.

        Args:
            config (Dict[str, Any]): 設定辞書
            logger (logging.Logger): ロガーインスタンス

        Returns:
            bool: バリデーション成功時True、失敗時False
        """
        epochs = config.get("epochs")

        # 必須チェック
        if epochs is None:
            logger.error(
                "epochs が設定されていません。configs/pochi_config.py で "
                "エポック数を設定してください。"
            )
            return False

        # 型チェック（int のみ受け入れ、boolは除外）
        if not isinstance(epochs, int) or isinstance(epochs, bool):
            logger.error(
                f"epochs は整数である必要があります。現在の型: {type(epochs).__name__}, "
                f"現在の値: {epochs}"
            )
            return False

        # 正の整数チェック（0は許可しない）
        if epochs <= 0:
            logger.error(f"epochs は正の整数である必要があります。現在の値: {epochs}")
            return False

        return True

    def _validate_batch_size(
        self, config: Dict[str, Any], logger: logging.Logger
    ) -> bool:
        """
        batch_sizeパラメータのバリデーション.

        Args:
            config (Dict[str, Any]): 設定辞書
            logger (logging.Logger): ロガーインスタンス

        Returns:
            bool: バリデーション成功時True、失敗時False
        """
        batch_size = config.get("batch_size")

        # 必須チェック
        if batch_size is None:
            logger.error(
                "batch_size が設定されていません。configs/pochi_config.py で "
                "バッチサイズを設定してください。"
            )
            return False

        # 型チェック（int のみ受け入れ、boolは除外）
        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            logger.error(
                f"batch_size は整数である必要があります。現在の型: {type(batch_size).__name__}, "
                f"現在の値: {batch_size}"
            )
            return False

        # 正の整数チェック（0は許可しない）
        if batch_size <= 0:
            logger.error(
                f"batch_size は正の整数である必要があります。現在の値: {batch_size}"
            )
            return False

        return True

    def _validate_model_name(
        self, config: Dict[str, Any], logger: logging.Logger
    ) -> bool:
        """
        model_nameパラメータのバリデーション.

        Args:
            config (Dict[str, Any]): 設定辞書
            logger (logging.Logger): ロガーインスタンス

        Returns:
            bool: バリデーション成功時True、失敗時False
        """
        model_name = config.get("model_name")

        # 必須チェック
        if model_name is None:
            logger.error(
                "model_name が設定されていません。configs/pochi_config.py で "
                "モデル名を設定してください。"
            )
            return False

        # 型チェック
        if not isinstance(model_name, str):
            logger.error(
                f"model_name は文字列である必要があります。現在の型: {type(model_name).__name__}, "
                f"現在の値: {model_name}"
            )
            return False

        # サポートされているモデル名チェック
        if model_name not in self.supported_models:
            logger.error(
                f"サポートされていないモデル名です: {model_name}. "
                f"サポート対象: {self.supported_models}"
            )
            return False

        return True
