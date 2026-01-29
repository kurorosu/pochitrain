"""EarlyStoppingValidatorのテスト."""

import pytest

from pochitrain.validation.validators.early_stopping_validator import (
    EarlyStoppingValidator,
)


@pytest.fixture
def validator():
    """EarlyStoppingValidatorのfixture."""
    return EarlyStoppingValidator()


class TestEarlyStoppingNotConfigured:
    """early_stopping設定が存在しない場合のテスト."""

    def test_no_config_success(self, validator, mocker):
        """early_stopping設定がない場合はバリデーション成功."""
        mock_logger = mocker.Mock()
        config = {"epochs": 50}

        result = validator.validate(config, mock_logger)

        assert result is True

    def test_disabled_config_success(self, validator, mocker):
        """enabled=Falseの場合はバリデーション成功."""
        mock_logger = mocker.Mock()
        config = {"early_stopping": {"enabled": False}}

        result = validator.validate(config, mock_logger)

        assert result is True


class TestEarlyStoppingTypeValidation:
    """early_stopping設定の型バリデーションテスト."""

    def test_non_dict_failure(self, validator, mocker):
        """辞書でない場合バリデーション失敗."""
        mock_logger = mocker.Mock()
        config = {"early_stopping": "invalid"}

        result = validator.validate(config, mock_logger)

        assert result is False
        mock_logger.error.assert_called_with(
            "early_stopping は辞書である必要があります。現在の型: str"
        )

    def test_enabled_non_bool_failure(self, validator, mocker):
        """enabledがboolでない場合バリデーション失敗."""
        mock_logger = mocker.Mock()
        config = {"early_stopping": {"enabled": "true"}}

        result = validator.validate(config, mock_logger)

        assert result is False
        mock_logger.error.assert_called_with(
            "early_stopping.enabled はboolである必要があります。"
            "現在の型: str, 現在の値: true"
        )


class TestPatienceValidation:
    """patienceパラメータのバリデーションテスト."""

    def test_patience_non_int_failure(self, validator, mocker):
        """patienceが整数でない場合バリデーション失敗."""
        mock_logger = mocker.Mock()
        config = {"early_stopping": {"enabled": True, "patience": 5.0}}

        result = validator.validate(config, mock_logger)

        assert result is False
        mock_logger.error.assert_called_with(
            "early_stopping.patience は整数である必要があります。"
            "現在の型: float, 現在の値: 5.0"
        )

    def test_patience_bool_failure(self, validator, mocker):
        """patienceがboolの場合バリデーション失敗."""
        mock_logger = mocker.Mock()
        config = {"early_stopping": {"enabled": True, "patience": True}}

        result = validator.validate(config, mock_logger)

        assert result is False
        mock_logger.error.assert_called_with(
            "early_stopping.patience は整数である必要があります。"
            "現在の型: bool, 現在の値: True"
        )

    def test_patience_zero_failure(self, validator, mocker):
        """patience=0の場合バリデーション失敗."""
        mock_logger = mocker.Mock()
        config = {"early_stopping": {"enabled": True, "patience": 0}}

        result = validator.validate(config, mock_logger)

        assert result is False
        mock_logger.error.assert_called_with(
            "early_stopping.patience は正の整数である必要があります。現在の値: 0"
        )

    def test_patience_negative_failure(self, validator, mocker):
        """patienceが負の場合バリデーション失敗."""
        mock_logger = mocker.Mock()
        config = {"early_stopping": {"enabled": True, "patience": -5}}

        result = validator.validate(config, mock_logger)

        assert result is False
        mock_logger.error.assert_called_with(
            "early_stopping.patience は正の整数である必要があります。現在の値: -5"
        )

    def test_patience_valid_success(self, validator, mocker):
        """有効なpatienceでバリデーション成功."""
        mock_logger = mocker.Mock()
        config = {"early_stopping": {"enabled": True, "patience": 10}}

        result = validator.validate(config, mock_logger)

        assert result is True


class TestMinDeltaValidation:
    """min_deltaパラメータのバリデーションテスト."""

    def test_min_delta_non_numeric_failure(self, validator, mocker):
        """min_deltaが数値でない場合バリデーション失敗."""
        mock_logger = mocker.Mock()
        config = {"early_stopping": {"enabled": True, "min_delta": "0.01"}}

        result = validator.validate(config, mock_logger)

        assert result is False
        mock_logger.error.assert_called_with(
            "early_stopping.min_delta は数値である必要があります。"
            "現在の型: str, 現在の値: 0.01"
        )

    def test_min_delta_bool_failure(self, validator, mocker):
        """min_deltaがboolの場合バリデーション失敗."""
        mock_logger = mocker.Mock()
        config = {"early_stopping": {"enabled": True, "min_delta": True}}

        result = validator.validate(config, mock_logger)

        assert result is False
        mock_logger.error.assert_called_with(
            "early_stopping.min_delta は数値である必要があります。"
            "現在の型: bool, 現在の値: True"
        )

    def test_min_delta_negative_failure(self, validator, mocker):
        """min_deltaが負の場合バリデーション失敗."""
        mock_logger = mocker.Mock()
        config = {"early_stopping": {"enabled": True, "min_delta": -0.1}}

        result = validator.validate(config, mock_logger)

        assert result is False
        mock_logger.error.assert_called_with(
            "early_stopping.min_delta は0以上である必要があります。現在の値: -0.1"
        )

    def test_min_delta_zero_success(self, validator, mocker):
        """min_delta=0でバリデーション成功."""
        mock_logger = mocker.Mock()
        config = {"early_stopping": {"enabled": True, "min_delta": 0.0}}

        result = validator.validate(config, mock_logger)

        assert result is True

    def test_min_delta_int_success(self, validator, mocker):
        """min_deltaが整数でもバリデーション成功."""
        mock_logger = mocker.Mock()
        config = {"early_stopping": {"enabled": True, "min_delta": 1}}

        result = validator.validate(config, mock_logger)

        assert result is True


class TestMonitorValidation:
    """monitorパラメータのバリデーションテスト."""

    def test_monitor_non_string_failure(self, validator, mocker):
        """monitorが文字列でない場合バリデーション失敗."""
        mock_logger = mocker.Mock()
        config = {"early_stopping": {"enabled": True, "monitor": 123}}

        result = validator.validate(config, mock_logger)

        assert result is False
        mock_logger.error.assert_called_with(
            "early_stopping.monitor は文字列である必要があります。"
            "現在の型: int, 現在の値: 123"
        )

    def test_monitor_unsupported_failure(self, validator, mocker):
        """サポートされていないmonitorでバリデーション失敗."""
        mock_logger = mocker.Mock()
        config = {"early_stopping": {"enabled": True, "monitor": "train_loss"}}

        result = validator.validate(config, mock_logger)

        assert result is False
        mock_logger.error.assert_called_with(
            "サポートされていないmonitor値です: train_loss. "
            "サポート対象: ['val_accuracy', 'val_loss']"
        )

    def test_monitor_val_accuracy_success(self, validator, mocker):
        """monitor='val_accuracy'でバリデーション成功."""
        mock_logger = mocker.Mock()
        config = {"early_stopping": {"enabled": True, "monitor": "val_accuracy"}}

        result = validator.validate(config, mock_logger)

        assert result is True

    def test_monitor_val_loss_success(self, validator, mocker):
        """monitor='val_loss'でバリデーション成功."""
        mock_logger = mocker.Mock()
        config = {"early_stopping": {"enabled": True, "monitor": "val_loss"}}

        result = validator.validate(config, mock_logger)

        assert result is True


class TestEarlyStoppingValidatorIntegration:
    """EarlyStoppingValidator統合テスト."""

    def test_full_valid_config_success(self, validator, mocker):
        """全パラメータが有効な場合バリデーション成功."""
        mock_logger = mocker.Mock()
        config = {
            "early_stopping": {
                "enabled": True,
                "patience": 10,
                "min_delta": 0.01,
                "monitor": "val_accuracy",
            }
        }

        result = validator.validate(config, mock_logger)

        assert result is True
        mock_logger.info.assert_called_with(
            "Early Stopping: 有効 "
            "(patience=10, min_delta=0.01, monitor=val_accuracy)"
        )

    def test_default_values_success(self, validator, mocker):
        """デフォルト値でバリデーション成功."""
        mock_logger = mocker.Mock()
        config = {"early_stopping": {"enabled": True}}

        result = validator.validate(config, mock_logger)

        assert result is True
        mock_logger.info.assert_called_with(
            "Early Stopping: 有効 " "(patience=10, min_delta=0.0, monitor=val_accuracy)"
        )
