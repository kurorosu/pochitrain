"""
pochitrain.validation: 設定バリデーションモジュール.

意図しない動作を防止するための設定チェック機能を提供します。
"""

from .config_validator import ConfigValidator

__all__ = ["ConfigValidator"]
