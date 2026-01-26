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
                "device設定が必須です。configs/pochi_train_config.pyで"
                "deviceを'cuda'または'cpu'に設定してください。"
            )
            logger.error("例: device = 'cuda' または device = 'cpu'")
            return False

        # CPU使用時の明示的な警告（意図しないパフォーマンス低下を防止）
        if device_config == "cpu":
            logger.warning("⚠️  CPU使用モードで実行中です")
            logger.warning("⚠️  GPU使用を推奨します（大幅な性能向上が期待できます）")
            logger.warning("⚠️  GPU使用時: device = 'cuda' に設定してください")

        # cudnn_benchmarkのバリデーション
        if not self._validate_cudnn_benchmark(config, logger, device_config):
            return False

        return True

    def _validate_cudnn_benchmark(
        self, config: Dict[str, Any], logger: logging.Logger, device_config: str
    ) -> bool:
        """
        cudnn_benchmark設定のバリデーション.

        Args:
            config (Dict[str, Any]): 設定辞書
            logger (logging.Logger): ロガーインスタンス
            device_config (str): デバイス設定

        Returns:
            bool: バリデーション成功時True、失敗時False
        """
        cudnn_benchmark = config.get("cudnn_benchmark")

        # 設定がない場合はスキップ（オプション設定）
        if cudnn_benchmark is None:
            return True

        # 型チェック（boolのみ許可）
        if not isinstance(cudnn_benchmark, bool):
            logger.error(
                f"cudnn_benchmark はbool型である必要があります。"
                f"現在の型: {type(cudnn_benchmark).__name__}, "
                f"現在の値: {cudnn_benchmark}"
            )
            return False

        # CPU使用時にcudnn_benchmark=Trueの場合は警告
        if device_config == "cpu" and cudnn_benchmark:
            logger.warning(
                "⚠️  cudnn_benchmark=Trueが設定されていますが、"
                "CPU使用時は無効です（GPUでのみ有効）"
            )

        return True
