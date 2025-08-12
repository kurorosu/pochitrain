"""
設定ファイルの統合バリデーター.

個別バリデーターを組み合わせて、設定全体をチェックします。
"""

import logging
from typing import Any, Dict

from .base_validator import BaseValidator
from .validators import (
    ClassWeightsValidator,
    DataValidator,
    DeviceValidator,
    OptimizerValidator,
    SchedulerValidator,
    TrainingValidator,
    TransformValidator,
)


class ConfigValidator:
    """設定ファイルの統合バリデーションクラス."""

    def __init__(self, logger: logging.Logger):
        """
        ConfigValidatorの初期化.

        Args:
            logger (logging.Logger): ロガーインスタンス
        """
        self.logger = logger
        self.class_weights_validator = ClassWeightsValidator()
        self.data_validator = DataValidator()
        self.device_validator = DeviceValidator()
        self.optimizer_validator = OptimizerValidator()
        self.scheduler_validator = SchedulerValidator()
        self.training_validator = TrainingValidator()
        self.transform_validator = TransformValidator()

    def validate(self, config: Dict[str, Any]) -> bool:
        """
        設定全体のバリデーション.

        Args:
            config (Dict[str, Any]): 設定辞書

        Returns:
            bool: 全てのバリデーションが成功した場合True、いずれかが失敗した場合False
        """
        # 個別バリデーターを順次実行
        validators: list[BaseValidator] = [
            self.data_validator,  # データパス（最初にチェック）
            self.class_weights_validator,  # クラス重み設定
            self.transform_validator,  # Transform設定
            self.device_validator,  # デバイス設定
            self.training_validator,  # 訓練設定（epochs、batch_size、model_name）
            self.optimizer_validator,  # 最適化器設定
            self.scheduler_validator,  # スケジューラー設定
        ]

        for validator in validators:
            if not validator.validate(config, self.logger):
                return False

        return True
