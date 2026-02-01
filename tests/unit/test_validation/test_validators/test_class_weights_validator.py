"""
ClassWeightsValidatorのテスト.
"""

import unittest
from unittest.mock import Mock

from pochitrain.validation.validators.class_weights_validator import (
    ClassWeightsValidator,
)


def assert_info_or_debug_called_with(mock_logger, message):
    """INFO/DEBUG のどちらかでメッセージが出ていることを確認する."""
    info_calls = mock_logger.info.call_args_list
    debug_calls = mock_logger.debug.call_args_list
    assert any(
        call.args and call.args[0] == message for call in info_calls + debug_calls
    )


class TestClassWeightsValidator(unittest.TestCase):
    """ClassWeightsValidatorのテストクラス."""

    def setUp(self):
        """テストの前処理."""
        self.validator = ClassWeightsValidator()
        self.mock_logger = Mock()

    def test_num_classes_missing_validation_fails(self):
        """num_classesが設定されていない場合はバリデーションが失敗することをテスト."""
        config = {"class_weights": [1.0, 2.0]}

        result = self.validator.validate(config, self.mock_logger)

        # アサーション
        assert result is False
        self.mock_logger.error.assert_called_once_with(
            "num_classes が設定されていません。"
            "configs/pochi_train_config.py で設定してください。"
        )

    def test_num_classes_none_validation_fails(self):
        """num_classesがNoneの場合はバリデーションが失敗することをテスト."""
        config = {"num_classes": None, "class_weights": [1.0, 2.0]}

        result = self.validator.validate(config, self.mock_logger)

        # アサーション
        assert result is False
        self.mock_logger.error.assert_called_once_with(
            "num_classes が設定されていません。"
            "configs/pochi_train_config.py で設定してください。"
        )

    def test_num_classes_invalid_type_validation_fails(self):
        """num_classesが整数でない場合はバリデーションが失敗することをテスト."""
        config = {"num_classes": "4", "class_weights": [1.0, 2.0]}

        result = self.validator.validate(config, self.mock_logger)

        # アサーション
        assert result is False
        self.mock_logger.error.assert_called_once_with(
            "num_classes は int である必要があります。" "現在の型: str, 現在の値: 4"
        )

    def test_num_classes_negative_validation_fails(self):
        """num_classesが負の値の場合はバリデーションが失敗することをテスト."""
        config = {"num_classes": -1, "class_weights": [1.0, 2.0]}

        result = self.validator.validate(config, self.mock_logger)

        # アサーション
        assert result is False
        self.mock_logger.error.assert_called_once_with(
            "num_classes は正の値である必要があります。現在の値: -1"
        )

    def test_num_classes_zero_validation_fails(self):
        """num_classesが0の場合はバリデーションが失敗することをテスト."""
        config = {"num_classes": 0, "class_weights": [1.0, 2.0]}

        result = self.validator.validate(config, self.mock_logger)

        # アサーション
        assert result is False
        self.mock_logger.error.assert_called_once_with(
            "num_classes は正の値である必要があります。現在の値: 0"
        )

    def test_class_weights_none_validation_succeeds(self):
        """class_weightsがNoneの場合はバリデーションが成功することをテスト."""
        config = {"num_classes": 4, "class_weights": None}

        result = self.validator.validate(config, self.mock_logger)

        # アサーション
        assert result is True
        assert_info_or_debug_called_with(
            self.mock_logger, "クラス重み: なし（均等扱い）"
        )

    def test_class_weights_invalid_type_validation_fails(self):
        """class_weightsがリスト/タプル以外の場合はバリデーションが失敗することをテスト."""
        config = {"num_classes": 4, "class_weights": "invalid"}

        result = self.validator.validate(config, self.mock_logger)

        # アサーション
        assert result is False
        self.mock_logger.error.assert_called_once_with(
            "class_weights はリスト形式で設定してください。現在の型: <class 'str'>"
        )

    def test_class_weights_length_mismatch_validation_fails(self):
        """class_weightsの要素数がnum_classesと一致しない場合はバリデーションが失敗することをテスト."""
        config = {"num_classes": 4, "class_weights": [1.0, 2.0]}  # 2要素、4クラス

        result = self.validator.validate(config, self.mock_logger)

        # アサーション
        assert result is False
        self.mock_logger.error.assert_called_once_with(
            "class_weights の要素数がnum_classesと一致しません。"
            "class_weights: 2要素, num_classes: 4"
        )

    def test_class_weights_invalid_element_type_validation_fails(self):
        """class_weightsの要素に数値以外が含まれる場合はバリデーションが失敗することをテスト."""
        config = {"num_classes": 3, "class_weights": [1.0, "invalid", 3.0]}

        result = self.validator.validate(config, self.mock_logger)

        # アサーション
        assert result is False
        self.mock_logger.error.assert_called_once_with(
            "class_weights[1] は数値である必要があります。"
            "現在の値: invalid (型: <class 'str'>)"
        )

    def test_class_weights_negative_value_validation_fails(self):
        """class_weightsの要素に負の値が含まれる場合はバリデーションが失敗することをテスト."""
        config = {"num_classes": 3, "class_weights": [1.0, -2.0, 3.0]}

        result = self.validator.validate(config, self.mock_logger)

        # アサーション
        assert result is False
        self.mock_logger.error.assert_called_once_with(
            "class_weights[1] は正の値である必要があります。現在の値: -2.0"
        )

    def test_class_weights_zero_value_validation_fails(self):
        """class_weightsの要素に0が含まれる場合はバリデーションが失敗することをテスト."""
        config = {"num_classes": 3, "class_weights": [1.0, 0.0, 3.0]}

        result = self.validator.validate(config, self.mock_logger)

        # アサーション
        assert result is False
        self.mock_logger.error.assert_called_once_with(
            "class_weights[1] は正の値である必要があります。現在の値: 0.0"
        )

    def test_class_weights_list_valid_validation_succeeds(self):
        """class_weightsがリスト形式で有効な場合はバリデーションが成功することをテスト."""
        config = {"num_classes": 3, "class_weights": [1.0, 2.5, 0.8]}

        result = self.validator.validate(config, self.mock_logger)

        # アサーション
        assert result is True
        assert_info_or_debug_called_with(
            self.mock_logger, "クラス重み: [1.0, 2.5, 0.8]"
        )

    def test_class_weights_tuple_valid_validation_succeeds(self):
        """class_weightsがタプル形式で有効な場合はバリデーションが成功することをテスト."""
        config = {"num_classes": 4, "class_weights": (1.0, 2.0, 1.5, 3.0)}

        result = self.validator.validate(config, self.mock_logger)

        # アサーション
        assert result is True
        assert_info_or_debug_called_with(
            self.mock_logger, "クラス重み: [1.0, 2.0, 1.5, 3.0]"
        )

    def test_class_weights_integer_values_validation_succeeds(self):
        """class_weightsが整数値で有効な場合はバリデーションが成功することをテスト."""
        config = {"num_classes": 2, "class_weights": [1, 3]}

        result = self.validator.validate(config, self.mock_logger)

        # アサーション
        assert result is True
        assert_info_or_debug_called_with(self.mock_logger, "クラス重み: [1, 3]")

    def test_class_weights_mixed_number_types_validation_succeeds(self):
        """class_weightsが整数と浮動小数点の混合で有効な場合はバリデーションが成功することをテスト."""
        config = {"num_classes": 3, "class_weights": [1, 2.5, 3]}

        result = self.validator.validate(config, self.mock_logger)

        # アサーション
        assert result is True
        assert_info_or_debug_called_with(self.mock_logger, "クラス重み: [1, 2.5, 3]")


if __name__ == "__main__":
    unittest.main()
