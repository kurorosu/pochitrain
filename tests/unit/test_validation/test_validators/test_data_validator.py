"""
DataValidatorのテスト.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from pochitrain.validation.validators.data_validator import DataValidator


class TestDataValidator(unittest.TestCase):
    """DataValidatorのテストクラス."""

    def setUp(self):
        """テストの前処理."""
        self.validator = DataValidator()
        self.mock_logger = Mock()

        # 一時ディレクトリを作成
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        # 有効なデータパスを作成
        self.valid_train_path = self.temp_path / "train"
        self.valid_val_path = self.temp_path / "val"
        self.valid_train_path.mkdir()
        self.valid_val_path.mkdir()

    def test_train_data_root_none_validation_fails(self):
        """train_data_root設定がNoneの場合はバリデーションが失敗することをテスト."""
        config = {"train_data_root": None, "val_data_root": str(self.valid_val_path)}

        result = self.validator.validate(config, self.mock_logger)

        # アサーション
        assert result is False
        self.mock_logger.error.assert_called_once_with(
            "train_data_root が必須です。configs/pochi_config.py で "
            "有効な訓練データパスを設定してください。"
        )

    def test_train_data_root_missing_validation_fails(self):
        """train_data_root設定が設定辞書にない場合はバリデーションが失敗することをテスト."""
        config = {"val_data_root": str(self.valid_val_path)}

        result = self.validator.validate(config, self.mock_logger)

        # アサーション
        assert result is False
        self.mock_logger.error.assert_called_once_with(
            "train_data_root が必須です。configs/pochi_config.py で "
            "有効な訓練データパスを設定してください。"
        )

    def test_train_data_root_not_exists_validation_fails(self):
        """train_data_rootが存在しない場合はバリデーションが失敗することをテスト."""
        nonexistent_path = str(self.temp_path / "nonexistent_train")
        config = {
            "train_data_root": nonexistent_path,
            "val_data_root": str(self.valid_val_path),
        }

        result = self.validator.validate(config, self.mock_logger)

        # アサーション
        assert result is False
        self.mock_logger.error.assert_called_once_with(
            f"訓練データパスが存在しません: {nonexistent_path}"
        )

    def test_val_data_root_none_validation_fails(self):
        """val_data_root設定がNoneの場合はバリデーションが失敗することをテスト."""
        config = {"train_data_root": str(self.valid_train_path), "val_data_root": None}

        result = self.validator.validate(config, self.mock_logger)

        # アサーション
        assert result is False
        self.mock_logger.error.assert_called_once_with(
            "val_data_root が必須です。configs/pochi_config.py で "
            "有効な検証データパスを設定してください。"
        )

    def test_val_data_root_not_exists_validation_fails(self):
        """val_data_rootが存在しない場合はバリデーションが失敗することをテスト."""
        nonexistent_path = str(self.temp_path / "nonexistent_val")
        config = {
            "train_data_root": str(self.valid_train_path),
            "val_data_root": nonexistent_path,
        }

        result = self.validator.validate(config, self.mock_logger)

        # アサーション
        assert result is False
        self.mock_logger.error.assert_called_once_with(
            f"検証データパスが存在しません: {nonexistent_path}"
        )

    def test_both_data_paths_valid(self):
        """両方のデータパス設定が有効な場合はバリデーションが成功することをテスト."""
        config = {
            "train_data_root": str(self.valid_train_path),
            "val_data_root": str(self.valid_val_path),
        }

        result = self.validator.validate(config, self.mock_logger)

        # アサーション
        assert result is True
        self.mock_logger.error.assert_not_called()

    def test_empty_string_paths_validation_fails(self):
        """空文字列のパス設定でバリデーションが失敗することをテスト."""
        config = {"train_data_root": "", "val_data_root": ""}

        result = self.validator.validate(config, self.mock_logger)

        # アサーション（train_data_rootで先に失敗）
        assert result is False
        self.mock_logger.error.assert_called_once_with(
            "train_data_root が必須です。configs/pochi_config.py で "
            "有効な訓練データパスを設定してください。"
        )


if __name__ == "__main__":
    unittest.main()
