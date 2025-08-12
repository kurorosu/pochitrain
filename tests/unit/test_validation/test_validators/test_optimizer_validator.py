"""OptimizerValidatorのテスト."""

import unittest
from unittest.mock import MagicMock

from pochitrain.validation.validators.optimizer_validator import OptimizerValidator


class TestOptimizerValidator(unittest.TestCase):
    """OptimizerValidatorのテストクラス."""

    def setUp(self):
        """テストの初期化."""
        self.validator = OptimizerValidator()
        self.mock_logger = MagicMock()

    def test_learning_rate_missing_failure(self):
        """learning_rate未設定でバリデーション失敗."""
        config = {"optimizer": "Adam"}

        result = self.validator.validate(config, self.mock_logger)

        assert result is False
        self.mock_logger.error.assert_called_with(
            "learning_rate が設定されていません。configs/pochi_config.py で "
            "学習率を設定してください。"
        )

    def test_learning_rate_invalid_type_failure(self):
        """learning_rateが数値でない場合バリデーション失敗."""
        config = {"learning_rate": "0.001", "optimizer": "Adam"}

        result = self.validator.validate(config, self.mock_logger)

        assert result is False
        self.mock_logger.error.assert_called_with(
            "learning_rate は数値である必要があります。現在の型: <class 'str'>"
        )

    def test_learning_rate_out_of_range_failure(self):
        """learning_rateが範囲外の場合バリデーション失敗."""
        # 0以下のケース
        config = {"learning_rate": 0, "optimizer": "Adam"}
        result = self.validator.validate(config, self.mock_logger)
        assert result is False

        # 1.0超過のケース
        config = {"learning_rate": 1.5, "optimizer": "Adam"}
        result = self.validator.validate(config, self.mock_logger)
        assert result is False

    def test_optimizer_missing_failure(self):
        """optimizer未設定でバリデーション失敗."""
        config = {"learning_rate": 0.001}

        result = self.validator.validate(config, self.mock_logger)

        assert result is False
        self.mock_logger.error.assert_called_with(
            "optimizer が設定されていません。configs/pochi_config.py で "
            "最適化器を設定してください。"
        )

    def test_optimizer_invalid_type_failure(self):
        """optimizerが文字列でない場合バリデーション失敗."""
        config = {"learning_rate": 0.001, "optimizer": 123}

        result = self.validator.validate(config, self.mock_logger)

        assert result is False
        self.mock_logger.error.assert_called_with(
            "optimizer は文字列である必要があります。現在の型: <class 'int'>"
        )

    def test_optimizer_unsupported_failure(self):
        """サポートされていないoptimizer名でバリデーション失敗."""
        config = {"learning_rate": 0.001, "optimizer": "RMSprop"}

        result = self.validator.validate(config, self.mock_logger)

        assert result is False
        self.mock_logger.error.assert_called_with(
            "サポートされていない最適化器です: RMSprop. "
            "サポート対象: ['Adam', 'SGD']"
        )

    def test_valid_adam_success(self):
        """有効なAdam設定でバリデーション成功."""
        config = {"learning_rate": 0.001, "optimizer": "Adam"}

        result = self.validator.validate(config, self.mock_logger)

        assert result is True
        self.mock_logger.info.assert_any_call("学習率: 0.001")
        self.mock_logger.info.assert_any_call("最適化器: Adam")

    def test_valid_sgd_success(self):
        """有効なSGD設定でバリデーション成功."""
        config = {"learning_rate": 0.1, "optimizer": "SGD"}

        result = self.validator.validate(config, self.mock_logger)

        assert result is True
        self.mock_logger.info.assert_any_call("学習率: 0.1")
        self.mock_logger.info.assert_any_call("最適化器: SGD")

    def test_learning_rate_boundary_values(self):
        """learning_rateの境界値テスト."""
        # 最小値（0に近い値）
        config = {"learning_rate": 0.0001, "optimizer": "Adam"}
        result = self.validator.validate(config, self.mock_logger)
        assert result is True

        # 最大値（1.0）
        config = {"learning_rate": 1.0, "optimizer": "Adam"}
        result = self.validator.validate(config, self.mock_logger)
        assert result is True


if __name__ == "__main__":
    unittest.main()
