"""
バリデーターの抽象基底クラス.

全てのバリデーターが実装すべきインターフェースを定義します。
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseValidator(ABC):
    """バリデーターの抽象基底クラス."""

    @abstractmethod
    def validate(self, config: Dict[str, Any], logger: logging.Logger) -> bool:
        """
        設定のバリデーションを実行する.

        Args:
            config (Dict[str, Any]): 設定辞書
            logger (logging.Logger): ロガーインスタンス

        Returns:
            bool: バリデーション成功時True、失敗時False
        """
        pass
