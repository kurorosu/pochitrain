"""ConfigValidator のテスト."""

import tempfile
from pathlib import Path
from unittest.mock import Mock

from pochitrain.validation.base_validator import BaseValidator
from pochitrain.validation.config_validator import ConfigValidator


def _build_valid_config(train_path: Path, val_path: Path) -> dict:
    """ConfigValidator 用の最小限の有効設定を生成する.

    Args:
        train_path (Path): 学習データのディレクトリパス.
        val_path (Path): 検証データのディレクトリパス.

    Returns:
        dict: ConfigValidator が通過する最小構成の設定辞書.
    """
    return {
        "device": "cuda",
        "num_classes": 4,
        "class_weights": None,
        "train_data_root": str(train_path),
        "val_data_root": str(val_path),
        "train_transform": "dummy_train_transform",
        "val_transform": "dummy_val_transform",
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "scheduler": None,
        "epochs": 10,
        "batch_size": 8,
        "model_name": "resnet18",
        "enable_layer_wise_lr": False,
    }


class TestConfigValidator:
    """ConfigValidator のテストクラス."""

    def setup_method(self):
        """一時的なデータセットディレクトリを作成する."""
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        self.valid_train_path = temp_path / "train"
        self.valid_val_path = temp_path / "val"
        self.valid_train_path.mkdir()
        self.valid_val_path.mkdir()

    def test_validation_success_with_real_validators(self):
        """実バリデータがすべて成功する場合に True を返すことを確認する."""
        logger = Mock()
        validator = ConfigValidator(logger)
        config = _build_valid_config(self.valid_train_path, self.valid_val_path)

        assert validator.validate(config) is True

    def test_validation_failure_when_data_path_missing(self):
        """DataValidator が失敗する場合に False を返すことを確認する."""
        logger = Mock()
        validator = ConfigValidator(logger)
        config = _build_valid_config(self.valid_train_path, self.valid_val_path)
        config["train_data_root"] = str(self.valid_train_path / "missing")

        assert validator.validate(config) is False

    def test_validation_failure_when_early_stopping_invalid(self):
        """EarlyStoppingValidator が失敗する場合に False を返すことを確認する."""
        logger = Mock()
        validator = ConfigValidator(logger)
        config = _build_valid_config(self.valid_train_path, self.valid_val_path)
        config["early_stopping"] = {"enabled": True, "patience": 0}

        assert validator.validate(config) is False

    def test_validation_failure_when_layer_wise_flag_missing(self):
        """層別学習率設定が欠落した場合に False を返すことを確認する."""
        logger = Mock()
        validator = ConfigValidator(logger)
        config = _build_valid_config(self.valid_train_path, self.valid_val_path)
        config.pop("enable_layer_wise_lr")

        assert validator.validate(config) is False

    def test_all_validators_inherit_base_validator(self):
        """すべてのバリデータメンバーが BaseValidator 継承であることを確認する."""
        logger = Mock()
        validator = ConfigValidator(logger)

        assert isinstance(validator.data_validator, BaseValidator)
        assert isinstance(validator.class_weights_validator, BaseValidator)
        assert isinstance(validator.transform_validator, BaseValidator)
        assert isinstance(validator.device_validator, BaseValidator)
        assert isinstance(validator.training_validator, BaseValidator)
        assert isinstance(validator.optimizer_validator, BaseValidator)
        assert isinstance(validator.scheduler_validator, BaseValidator)
        assert isinstance(validator.layer_wise_lr_validator, BaseValidator)
        assert isinstance(validator.early_stopping_validator, BaseValidator)
