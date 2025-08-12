"""ConfigValidatorのテスト."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from pochitrain.validation.config_validator import ConfigValidator


class TestConfigValidator(unittest.TestCase):
    """ConfigValidatorのテストクラス."""

    def setUp(self):
        """テストの初期化."""
        self.mock_logger = MagicMock()
        self.validator = ConfigValidator(self.mock_logger)

        # テスト用の一時ディレクトリ
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.valid_train_path = self.temp_path / "train"
        self.valid_val_path = self.temp_path / "val"
        self.valid_train_path.mkdir()
        self.valid_val_path.mkdir()

    @patch("pochitrain.validation.config_validator.SchedulerValidator")
    @patch("pochitrain.validation.config_validator.OptimizerValidator")
    @patch("pochitrain.validation.config_validator.DeviceValidator")
    @patch("pochitrain.validation.config_validator.TransformValidator")
    @patch("pochitrain.validation.config_validator.ClassWeightsValidator")
    @patch("pochitrain.validation.config_validator.DataValidator")
    def test_validation_success(
        self,
        mock_data_validator_class,
        mock_class_weights_validator_class,
        mock_transform_validator_class,
        mock_device_validator_class,
        mock_optimizer_validator_class,
        mock_scheduler_validator_class,
    ):
        """全てのバリデーションが成功する場合のテスト."""
        # モックの設定
        mock_data_validator = Mock()
        mock_data_validator.validate.return_value = True
        mock_data_validator_class.return_value = mock_data_validator

        mock_class_weights_validator = Mock()
        mock_class_weights_validator.validate.return_value = True
        mock_class_weights_validator_class.return_value = mock_class_weights_validator

        mock_transform_validator = Mock()
        mock_transform_validator.validate.return_value = True
        mock_transform_validator_class.return_value = mock_transform_validator

        mock_device_validator = Mock()
        mock_device_validator.validate.return_value = True
        mock_device_validator_class.return_value = mock_device_validator

        mock_optimizer_validator = Mock()
        mock_optimizer_validator.validate.return_value = True
        mock_optimizer_validator_class.return_value = mock_optimizer_validator

        mock_scheduler_validator = Mock()
        mock_scheduler_validator.validate.return_value = True
        mock_scheduler_validator_class.return_value = mock_scheduler_validator

        # テスト実行
        validator = ConfigValidator(self.mock_logger)
        config = {"device": "cuda"}
        result = validator.validate(config)

        # アサーション
        assert result is True
        mock_data_validator.validate.assert_called_once_with(config, self.mock_logger)
        mock_class_weights_validator.validate.assert_called_once_with(
            config, self.mock_logger
        )
        mock_transform_validator.validate.assert_called_once_with(
            config, self.mock_logger
        )
        mock_device_validator.validate.assert_called_once_with(config, self.mock_logger)
        mock_optimizer_validator.validate.assert_called_once_with(
            config, self.mock_logger
        )
        mock_scheduler_validator.validate.assert_called_once_with(
            config, self.mock_logger
        )

    @patch("pochitrain.validation.config_validator.SchedulerValidator")
    @patch("pochitrain.validation.config_validator.OptimizerValidator")
    @patch("pochitrain.validation.config_validator.DeviceValidator")
    @patch("pochitrain.validation.config_validator.TransformValidator")
    @patch("pochitrain.validation.config_validator.ClassWeightsValidator")
    @patch("pochitrain.validation.config_validator.DataValidator")
    def test_validation_failure(
        self,
        mock_data_validator_class,
        mock_class_weights_validator_class,
        mock_transform_validator_class,
        mock_device_validator_class,
        mock_optimizer_validator_class,
        mock_scheduler_validator_class,
    ):
        """いずれかのバリデーションが失敗する場合のテスト."""
        # モックの設定（DataValidatorが失敗）
        mock_data_validator = Mock()
        mock_data_validator.validate.return_value = False
        mock_data_validator_class.return_value = mock_data_validator

        mock_class_weights_validator = Mock()
        mock_class_weights_validator.validate.return_value = True
        mock_class_weights_validator_class.return_value = mock_class_weights_validator

        mock_transform_validator = Mock()
        mock_transform_validator.validate.return_value = True
        mock_transform_validator_class.return_value = mock_transform_validator

        mock_device_validator = Mock()
        mock_device_validator.validate.return_value = True
        mock_device_validator_class.return_value = mock_device_validator

        mock_optimizer_validator = Mock()
        mock_optimizer_validator.validate.return_value = True
        mock_optimizer_validator_class.return_value = mock_optimizer_validator

        mock_scheduler_validator = Mock()
        mock_scheduler_validator.validate.return_value = True
        mock_scheduler_validator_class.return_value = mock_scheduler_validator

        # テスト実行
        validator = ConfigValidator(self.mock_logger)
        config = {"device": "cuda"}
        result = validator.validate(config)

        # アサーション
        assert result is False
        mock_data_validator.validate.assert_called_once_with(config, self.mock_logger)
        # 最初のバリデーターが失敗したので、後続は実行されない
        mock_class_weights_validator.validate.assert_not_called()
        mock_transform_validator.validate.assert_not_called()
        mock_device_validator.validate.assert_not_called()
        mock_optimizer_validator.validate.assert_not_called()
        mock_scheduler_validator.validate.assert_not_called()

    def test_all_validators_inherit_base_validator(self):
        """全てのバリデーターがBaseValidatorを継承していることをテスト."""
        from pochitrain.validation.base_validator import BaseValidator

        assert isinstance(self.validator.data_validator, BaseValidator)
        assert isinstance(self.validator.class_weights_validator, BaseValidator)
        assert isinstance(self.validator.transform_validator, BaseValidator)
        assert isinstance(self.validator.device_validator, BaseValidator)
        assert isinstance(self.validator.optimizer_validator, BaseValidator)
        assert isinstance(self.validator.scheduler_validator, BaseValidator)

    def test_validation_with_real_validators(self):
        """実際のバリデーターを使った統合テスト."""
        # 成功ケース
        config_success = {
            "device": "cuda",
            "num_classes": 4,
            "class_weights": None,
            "train_data_root": str(self.valid_train_path),
            "val_data_root": str(self.valid_val_path),
            "train_transform": "valid_train_transform",
            "val_transform": "valid_val_transform",
            "learning_rate": 0.001,
            "optimizer": "Adam",
            "scheduler": "StepLR",
            "scheduler_params": {"step_size": 30, "gamma": 0.1},
        }
        result_success = self.validator.validate(config_success)
        assert result_success is True

        # 失敗ケース（device=None）
        config_failure_device = {
            "device": None,
            "num_classes": 4,
            "class_weights": None,
            "train_data_root": str(self.valid_train_path),
            "val_data_root": str(self.valid_val_path),
            "train_transform": "valid_train_transform",
            "val_transform": "valid_val_transform",
            "learning_rate": 0.001,
            "optimizer": "Adam",
            "scheduler": None,
        }
        result_failure_device = self.validator.validate(config_failure_device)
        assert result_failure_device is False

        # 失敗ケース（transform=None）
        config_failure_transform = {
            "device": "cuda",
            "num_classes": 4,
            "class_weights": None,
            "train_data_root": str(self.valid_train_path),
            "val_data_root": str(self.valid_val_path),
            "train_transform": None,
            "val_transform": "valid_val_transform",
            "learning_rate": 0.001,
            "optimizer": "Adam",
            "scheduler": None,
        }
        result_failure_transform = self.validator.validate(config_failure_transform)
        assert result_failure_transform is False

        # 失敗ケース（scheduler設定エラー）
        config_failure_scheduler = {
            "device": "cuda",
            "num_classes": 4,
            "class_weights": None,
            "train_data_root": str(self.valid_train_path),
            "val_data_root": str(self.valid_val_path),
            "train_transform": "valid_train_transform",
            "val_transform": "valid_val_transform",
            "learning_rate": 0.001,
            "optimizer": "Adam",
            "scheduler": "StepLR",
            "scheduler_params": None,  # scheduler_params がNone
        }
        result_failure_scheduler = self.validator.validate(config_failure_scheduler)
        assert result_failure_scheduler is False

        # 失敗ケース（optimizer設定エラー）
        config_failure_optimizer = {
            "device": "cuda",
            "num_classes": 4,
            "class_weights": None,
            "train_data_root": str(self.valid_train_path),
            "val_data_root": str(self.valid_val_path),
            "train_transform": "valid_train_transform",
            "val_transform": "valid_val_transform",
            "learning_rate": 0.001,
            "optimizer": "InvalidOptimizer",  # サポート外optimizer
            "scheduler": None,
        }
        result_failure_optimizer = self.validator.validate(config_failure_optimizer)
        assert result_failure_optimizer is False


if __name__ == "__main__":
    unittest.main()
