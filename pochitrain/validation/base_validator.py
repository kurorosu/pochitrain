"""
バリデーターの抽象基底クラス.

全てのバリデーターが実装すべきインターフェースを定義します。
共通のバリデーションヘルパーメソッドを提供します。
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence, Tuple, Type, Union


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

    def _validate_required_type(
        self,
        value: Any,
        field_name: str,
        expected_type: Union[Type[Any], Tuple[Type[Any], ...]],
        logger: logging.Logger,
        *,
        exclude_bool: bool = False,
    ) -> bool:
        """
        必須フィールドの存在チェックと型チェックを行う.

        Args:
            value: チェック対象の値
            field_name: フィールド名(エラーメッセージ用)
            expected_type: 期待する型(単一またはタプル)
            logger: ロガーインスタンス
            exclude_bool: Trueの場合、bool型を除外する

        Returns:
            bool: バリデーション成功時True、失敗時False
        """
        if value is None:
            logger.error(
                f"{field_name} が設定されていません。"
                f"configs/pochi_train_config.py で設定してください。"
            )
            return False

        if exclude_bool and isinstance(value, bool):
            logger.error(
                f"{field_name} は {self._type_name(expected_type)} である必要があります。"
                f"現在の型: {type(value).__name__}, 現在の値: {value}"
            )
            return False

        if not isinstance(value, expected_type):
            logger.error(
                f"{field_name} は {self._type_name(expected_type)} である必要があります。"
                f"現在の型: {type(value).__name__}, 現在の値: {value}"
            )
            return False

        return True

    def _validate_positive(
        self,
        value: Any,
        field_name: str,
        logger: logging.Logger,
    ) -> bool:
        """
        値が正の数であることをチェックする.

        Args:
            value: チェック対象の値
            field_name: フィールド名(エラーメッセージ用)
            logger: ロガーインスタンス

        Returns:
            bool: バリデーション成功時True、失敗時False
        """
        if value <= 0:
            logger.error(
                f"{field_name} は正の値である必要があります。現在の値: {value}"
            )
            return False
        return True

    def _validate_range(
        self,
        value: Any,
        field_name: str,
        logger: logging.Logger,
        *,
        gt: Optional[Union[int, float]] = None,
        ge: Optional[Union[int, float]] = None,
        lt: Optional[Union[int, float]] = None,
        le: Optional[Union[int, float]] = None,
    ) -> bool:
        """
        値が指定範囲内であることをチェックする.

        Args:
            value: チェック対象の値
            field_name: フィールド名(エラーメッセージ用)
            logger: ロガーインスタンス
            gt: より大きい(exclusive lower bound)
            ge: 以上(inclusive lower bound)
            lt: より小さい(exclusive upper bound)
            le: 以下(inclusive upper bound)

        Returns:
            bool: バリデーション成功時True、失敗時False
        """
        bounds = []
        if gt is not None:
            if value <= gt:
                bounds.append(f"{gt} < ")
        if ge is not None:
            if value < ge:
                bounds.append(f"{ge} <= ")
        if lt is not None:
            if value >= lt:
                bounds.append(f" < {lt}")
        if le is not None:
            if value > le:
                bounds.append(f" <= {le}")

        if bounds:
            range_parts = []
            if gt is not None:
                range_parts.append(f"{gt} < ")
            if ge is not None:
                range_parts.append(f"{ge} <= ")
            range_parts.append(field_name)
            if lt is not None:
                range_parts.append(f" < {lt}")
            if le is not None:
                range_parts.append(f" <= {le}")
            range_str = "".join(range_parts)

            logger.error(
                f"{field_name} は {range_str} の範囲である必要があります。"
                f"現在の値: {value}"
            )
            return False

        return True

    def _validate_choice(
        self,
        value: Any,
        field_name: str,
        choices: Sequence[Any],
        logger: logging.Logger,
    ) -> bool:
        """
        値が選択肢のいずれかであることをチェックする.

        Args:
            value: チェック対象の値
            field_name: フィールド名(エラーメッセージ用)
            choices: 有効な選択肢のシーケンス
            logger: ロガーインスタンス

        Returns:
            bool: バリデーション成功時True、失敗時False
        """
        if value not in choices:
            logger.error(
                f"サポートされていない{field_name}です: {value}. "
                f"サポート対象: {list(choices)}"
            )
            return False
        return True

    @staticmethod
    def _type_name(expected_type: Union[Type[Any], Tuple[Type[Any], ...]]) -> str:
        """型名の表示用文字列を生成する."""
        if isinstance(expected_type, tuple):
            return " または ".join(t.__name__ for t in expected_type)
        return expected_type.__name__
