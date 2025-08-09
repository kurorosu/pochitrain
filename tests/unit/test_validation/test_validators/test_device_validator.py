"""
DeviceValidatorのテスト.
"""

import unittest
from unittest.mock import Mock

from pochitrain.validation.validators.device_validator import DeviceValidator


class TestDeviceValidator(unittest.TestCase):
    """DeviceValidatorのテストクラス."""

    def setUp(self):
        """テストの前処理."""
        self.validator = DeviceValidator()
        self.mock_logger = Mock()

    def test_device_none_validation_fails(self):
        """device設定がNoneの場合はバリデーションが失敗することをテスト."""
        config = {"device": None}

        result = self.validator.validate(config, self.mock_logger)

        # アサーション
        assert result is False
        # エラーメッセージが出力されることを確認
        assert self.mock_logger.error.call_count == 2
        self.mock_logger.error.assert_any_call(
            "device設定が必須です。configs/pochi_config.pyでdeviceを'cuda'または'cpu'に設定してください。"
        )
        self.mock_logger.error.assert_any_call(
            "例: device = 'cuda' または device = 'cpu'"
        )

    def test_device_cpu_shows_warning(self):
        """device設定が'cpu'の場合は警告メッセージを表示することをテスト."""
        config = {"device": "cpu"}

        result = self.validator.validate(config, self.mock_logger)

        # アサーション
        assert result is True
        # 警告メッセージが出力されることを確認
        assert self.mock_logger.warning.call_count == 3
        self.mock_logger.warning.assert_any_call("⚠️  CPU使用モードで実行中です")
        self.mock_logger.warning.assert_any_call(
            "⚠️  GPU使用を推奨します（大幅な性能向上が期待できます）"
        )
        self.mock_logger.warning.assert_any_call(
            "⚠️  GPU使用時: device = 'cuda' に設定してください"
        )

    def test_device_cuda_no_warning(self):
        """device設定が'cuda'の場合は警告メッセージを表示しないことをテスト."""
        config = {"device": "cuda"}

        result = self.validator.validate(config, self.mock_logger)

        # アサーション
        assert result is True
        # 警告メッセージが出力されないことを確認
        self.mock_logger.warning.assert_not_called()

    def test_device_missing_from_config(self):
        """設定辞書にdeviceキーがない場合のテスト."""
        config = {}  # deviceキーなし

        result = self.validator.validate(config, self.mock_logger)

        # アサーション（device=Noneと同じ扱い）
        assert result is False
        assert self.mock_logger.error.call_count == 2


if __name__ == "__main__":
    unittest.main()
