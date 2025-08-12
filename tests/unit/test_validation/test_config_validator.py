"""ConfigValidatorのテスト."""

import tempfile
from pathlib import Path

from pochitrain.validation.config_validator import ConfigValidator


class TestConfigValidator:
    """ConfigValidatorのテストクラス."""

    def setup_method(self):
        """テストの初期化."""
        # テスト用の一時ディレクトリ
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.valid_train_path = self.temp_path / "train"
        self.valid_val_path = self.temp_path / "val"
        self.valid_train_path.mkdir()
        self.valid_val_path.mkdir()

    def test_validation_success(self, mocker):
        """全てのバリデーションが成功する場合のテスト."""
        mock_logger = mocker.MagicMock()

        # モックの設定
        mock_data_validator = mocker.Mock()
        mock_data_validator.validate.return_value = True
        mock_data_validator_class = mocker.patch(
            "pochitrain.validation.config_validator.DataValidator"
        )
        mock_data_validator_class.return_value = mock_data_validator

        mock_class_weights_validator = mocker.Mock()
        mock_class_weights_validator.validate.return_value = True
        mock_class_weights_validator_class = mocker.patch(
            "pochitrain.validation.config_validator.ClassWeightsValidator"
        )
        mock_class_weights_validator_class.return_value = mock_class_weights_validator

        mock_transform_validator = mocker.Mock()
        mock_transform_validator.validate.return_value = True
        mock_transform_validator_class = mocker.patch(
            "pochitrain.validation.config_validator.TransformValidator"
        )
        mock_transform_validator_class.return_value = mock_transform_validator

        mock_device_validator = mocker.Mock()
        mock_device_validator.validate.return_value = True
        mock_device_validator_class = mocker.patch(
            "pochitrain.validation.config_validator.DeviceValidator"
        )
        mock_device_validator_class.return_value = mock_device_validator

        mock_optimizer_validator = mocker.Mock()
        mock_optimizer_validator.validate.return_value = True
        mock_optimizer_validator_class = mocker.patch(
            "pochitrain.validation.config_validator.OptimizerValidator"
        )
        mock_optimizer_validator_class.return_value = mock_optimizer_validator

        mock_scheduler_validator = mocker.Mock()
        mock_scheduler_validator.validate.return_value = True
        mock_scheduler_validator_class = mocker.patch(
            "pochitrain.validation.config_validator.SchedulerValidator"
        )
        mock_scheduler_validator_class.return_value = mock_scheduler_validator

        mock_training_validator = mocker.Mock()
        mock_training_validator.validate.return_value = True
        mock_training_validator_class = mocker.patch(
            "pochitrain.validation.config_validator.TrainingValidator"
        )
        mock_training_validator_class.return_value = mock_training_validator

        # テスト実行
        validator = ConfigValidator(mock_logger)
        config = {"device": "cuda"}
        result = validator.validate(config)

        # アサーション
        assert result is True
        mock_data_validator.validate.assert_called_once_with(config, mock_logger)
        mock_class_weights_validator.validate.assert_called_once_with(
            config, mock_logger
        )
        mock_transform_validator.validate.assert_called_once_with(config, mock_logger)
        mock_device_validator.validate.assert_called_once_with(config, mock_logger)
        mock_training_validator.validate.assert_called_once_with(config, mock_logger)
        mock_optimizer_validator.validate.assert_called_once_with(config, mock_logger)
        mock_scheduler_validator.validate.assert_called_once_with(config, mock_logger)

    def test_validation_failure(self, mocker):
        """いずれかのバリデーションが失敗する場合のテスト."""
        mock_logger = mocker.MagicMock()

        # モックの設定（DataValidatorが失敗）
        mock_data_validator = mocker.Mock()
        mock_data_validator.validate.return_value = False
        mock_data_validator_class = mocker.patch(
            "pochitrain.validation.config_validator.DataValidator"
        )
        mock_data_validator_class.return_value = mock_data_validator

        mock_class_weights_validator = mocker.Mock()
        mock_class_weights_validator.validate.return_value = True
        mock_class_weights_validator_class = mocker.patch(
            "pochitrain.validation.config_validator.ClassWeightsValidator"
        )
        mock_class_weights_validator_class.return_value = mock_class_weights_validator

        mock_transform_validator = mocker.Mock()
        mock_transform_validator.validate.return_value = True
        mock_transform_validator_class = mocker.patch(
            "pochitrain.validation.config_validator.TransformValidator"
        )
        mock_transform_validator_class.return_value = mock_transform_validator

        mock_device_validator = mocker.Mock()
        mock_device_validator.validate.return_value = True
        mock_device_validator_class = mocker.patch(
            "pochitrain.validation.config_validator.DeviceValidator"
        )
        mock_device_validator_class.return_value = mock_device_validator

        mock_optimizer_validator = mocker.Mock()
        mock_optimizer_validator.validate.return_value = True
        mock_optimizer_validator_class = mocker.patch(
            "pochitrain.validation.config_validator.OptimizerValidator"
        )
        mock_optimizer_validator_class.return_value = mock_optimizer_validator

        mock_scheduler_validator = mocker.Mock()
        mock_scheduler_validator.validate.return_value = True
        mock_scheduler_validator_class = mocker.patch(
            "pochitrain.validation.config_validator.SchedulerValidator"
        )
        mock_scheduler_validator_class.return_value = mock_scheduler_validator

        mock_training_validator = mocker.Mock()
        mock_training_validator.validate.return_value = True
        mock_training_validator_class = mocker.patch(
            "pochitrain.validation.config_validator.TrainingValidator"
        )
        mock_training_validator_class.return_value = mock_training_validator

        # テスト実行
        validator = ConfigValidator(mock_logger)
        config = {"device": "cuda"}
        result = validator.validate(config)

        # アサーション
        assert result is False
        mock_data_validator.validate.assert_called_once_with(config, mock_logger)
        # 最初のバリデーターが失敗したので、後続は実行されない
        mock_class_weights_validator.validate.assert_not_called()
        mock_transform_validator.validate.assert_not_called()
        mock_device_validator.validate.assert_not_called()
        mock_training_validator.validate.assert_not_called()
        mock_optimizer_validator.validate.assert_not_called()
        mock_scheduler_validator.validate.assert_not_called()

    def test_all_validators_inherit_base_validator(self, mocker):
        """全てのバリデーターがBaseValidatorを継承していることをテスト."""
        from pochitrain.validation.base_validator import BaseValidator

        mock_logger = mocker.MagicMock()
        validator = ConfigValidator(mock_logger)

        assert isinstance(validator.data_validator, BaseValidator)
        assert isinstance(validator.class_weights_validator, BaseValidator)
        assert isinstance(validator.transform_validator, BaseValidator)
        assert isinstance(validator.device_validator, BaseValidator)
        assert isinstance(validator.training_validator, BaseValidator)
        assert isinstance(validator.optimizer_validator, BaseValidator)
        assert isinstance(validator.scheduler_validator, BaseValidator)

    def test_validation_with_real_validators(self, mocker):
        """実際のバリデーターを使った統合テスト."""
        mock_logger = mocker.MagicMock()
        validator = ConfigValidator(mock_logger)

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
            "epochs": 100,
            "batch_size": 32,
            "model_name": "resnet50",
        }
        result_success = validator.validate(config_success)
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
            "epochs": 100,
            "batch_size": 32,
            "model_name": "resnet50",
        }
        result_failure_device = validator.validate(config_failure_device)
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
            "epochs": 100,
            "batch_size": 32,
            "model_name": "resnet50",
        }
        result_failure_transform = validator.validate(config_failure_transform)
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
            "epochs": 100,
            "batch_size": 32,
            "model_name": "resnet50",
        }
        result_failure_scheduler = validator.validate(config_failure_scheduler)
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
            "epochs": 100,
            "batch_size": 32,
            "model_name": "resnet50",
        }
        result_failure_optimizer = validator.validate(config_failure_optimizer)
        assert result_failure_optimizer is False
