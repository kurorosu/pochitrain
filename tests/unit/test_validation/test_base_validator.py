"""
BaseValidatorのテスト.
"""

import pytest

from pochitrain.validation.base_validator import BaseValidator


class ConcreteValidator(BaseValidator):
    """テスト用の具象バリデータークラス."""

    def validate(self, config, logger):
        """テスト用のvalidate実装."""
        return True


def test_abstract_class_cannot_be_instantiated():
    """BaseValidatorは抽象クラスなので直接インスタンス化できないことをテスト."""
    with pytest.raises(TypeError):
        BaseValidator()


def test_concrete_validator_can_be_instantiated():
    """validateメソッドを実装した具象クラスはインスタンス化できることをテスト."""
    validator = ConcreteValidator()
    assert isinstance(validator, BaseValidator)


def test_concrete_validator_validates(mocker):
    """具象バリデーターのvalidateメソッドが正常に動作することをテスト."""
    validator = ConcreteValidator()
    mock_logger = mocker.Mock()
    config = {"test": "value"}

    result = validator.validate(config, mock_logger)

    assert result is True


def test_incomplete_validator_cannot_be_instantiated():
    """validateメソッドを実装していないクラスはインスタンス化できないことをテスト."""

    class IncompleteValidator(BaseValidator):
        """validateメソッドを実装していない不完全なバリデーター."""

        pass

    with pytest.raises(TypeError):
        IncompleteValidator()
