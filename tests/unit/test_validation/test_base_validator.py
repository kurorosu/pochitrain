"""
BaseValidatorのテスト.
"""

import unittest
from unittest.mock import Mock

from pochitrain.validation.base_validator import BaseValidator


class ConcreteValidator(BaseValidator):
    """テスト用の具象バリデータークラス."""

    def validate(self, config, logger):
        """テスト用のvalidate実装."""
        return True


class TestBaseValidator(unittest.TestCase):
    """BaseValidatorのテストクラス."""

    def test_abstract_class_cannot_be_instantiated(self):
        """BaseValidatorは抽象クラスなので直接インスタンス化できないことをテスト."""
        with self.assertRaises(TypeError):
            BaseValidator()

    def test_concrete_validator_can_be_instantiated(self):
        """validateメソッドを実装した具象クラスはインスタンス化できることをテスト."""
        validator = ConcreteValidator()
        self.assertIsInstance(validator, BaseValidator)

    def test_concrete_validator_validates(self):
        """具象バリデーターのvalidateメソッドが正常に動作することをテスト."""
        validator = ConcreteValidator()
        mock_logger = Mock()
        config = {"test": "value"}

        result = validator.validate(config, mock_logger)

        self.assertTrue(result)

    def test_incomplete_validator_cannot_be_instantiated(self):
        """validateメソッドを実装していないクラスはインスタンス化できないことをテスト."""

        class IncompleteValidator(BaseValidator):
            """validateメソッドを実装していない不完全なバリデーター."""

            pass

        with self.assertRaises(TypeError):
            IncompleteValidator()


if __name__ == "__main__":
    unittest.main()
