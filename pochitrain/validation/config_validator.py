"""
設定ファイルの統合バリデーター.

個別バリデーターを組み合わせて、設定全体をチェックします。
"""

import logging
from typing import Any, Dict, Protocol

from .validators import (
    DataValidator,
    DeviceValidator,
    SchedulerValidator,
    TransformValidator,
)


class ValidatorProtocol(Protocol):
    """バリデータープロトコル."""

    def validate(self, config: Dict[str, Any], logger: logging.Logger) -> bool:
        """バリデーション実行."""
        ...


class ConfigValidator:
    """設定ファイルの統合バリデーションクラス."""

    def __init__(self, logger: logging.Logger):
        """
        ConfigValidatorの初期化.

        Args:
            logger (logging.Logger): ロガーインスタンス
        """
        self.logger = logger
        self.device_validator = DeviceValidator()
        self.transform_validator = TransformValidator()
        self.data_validator = DataValidator()
        self.scheduler_validator = SchedulerValidator()

    def validate(self, config: Dict[str, Any]) -> bool:
        """
        設定全体のバリデーション.

        Args:
            config (Dict[str, Any]): 設定辞書

        Returns:
            bool: 全てのバリデーションが成功した場合True、いずれかが失敗した場合False
        """
        # 個別バリデーターを順次実行
        validators: list[ValidatorProtocol] = [
            self.data_validator,  # データパス（最初にチェック）
            self.transform_validator,  # Transform設定
            self.device_validator,  # デバイス設定
            self.scheduler_validator,  # スケジューラー設定
        ]

        for validator in validators:
            if not validator.validate(config, self.logger):
                return False

        return True
