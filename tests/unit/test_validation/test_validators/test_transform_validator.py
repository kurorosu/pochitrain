"""
TransformValidatorのテスト.
"""

import unittest
from unittest.mock import Mock

from pochitrain.validation.validators.transform_validator import TransformValidator


class TestTransformValidator(unittest.TestCase):
    """TransformValidatorのテストクラス."""

    def setUp(self):
        """テストの前処理."""
        self.validator = TransformValidator()
        self.mock_logger = Mock()

    def test_train_transform_none_validation_fails(self):
        """train_transform設定がNoneの場合はバリデーションが失敗することをテスト."""
        config = {"train_transform": None, "val_transform": "dummy_transform"}

        result = self.validator.validate(config, self.mock_logger)

        # アサーション
        assert result is False
        self.mock_logger.error.assert_called_once_with(
            "train_transform が必須です。configs/pochi_config.py で "
            "transforms.Compose([...]) を train_transform として定義してください。"
        )

    def test_val_transform_none_validation_fails(self):
        """val_transform設定がNoneの場合はバリデーションが失敗することをテスト."""
        config = {"train_transform": "dummy_transform", "val_transform": None}

        result = self.validator.validate(config, self.mock_logger)

        # アサーション
        assert result is False
        self.mock_logger.error.assert_called_once_with(
            "val_transform が必須です。configs/pochi_config.py で "
            "transforms.Compose([...]) を val_transform として定義してください。"
        )

    def test_both_transforms_none_validation_fails(self):
        """両方のtransform設定がNoneの場合はバリデーションが失敗することをテスト."""
        config = {"train_transform": None, "val_transform": None}

        result = self.validator.validate(config, self.mock_logger)

        # アサーション（train_transformで先に失敗）
        assert result is False
        self.mock_logger.error.assert_called_once_with(
            "train_transform が必須です。configs/pochi_config.py で "
            "transforms.Compose([...]) を train_transform として定義してください。"
        )

    def test_transforms_missing_from_config(self):
        """設定辞書にtransformキーがない場合のテスト."""
        config = {}  # transformキーなし

        result = self.validator.validate(config, self.mock_logger)

        # アサーション（train_transform=Noneと同じ扱い）
        assert result is False
        self.mock_logger.error.assert_called_once_with(
            "train_transform が必須です。configs/pochi_config.py で "
            "transforms.Compose([...]) を train_transform として定義してください。"
        )

    def test_both_transforms_valid(self):
        """両方のtransform設定が有効な場合はバリデーションが成功することをテスト."""
        config = {
            "train_transform": "valid_train_transform",
            "val_transform": "valid_val_transform",
        }

        result = self.validator.validate(config, self.mock_logger)

        # アサーション
        assert result is True
        self.mock_logger.error.assert_not_called()


if __name__ == "__main__":
    unittest.main()
