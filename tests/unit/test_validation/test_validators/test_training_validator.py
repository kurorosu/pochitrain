"""TrainingValidatorのテスト."""

import pytest

from pochitrain.validation.validators.training_validator import TrainingValidator


@pytest.fixture
def validator():
    """TrainingValidatorのfixture."""
    return TrainingValidator()


class TestEpochsValidation:
    """epochsパラメータのバリデーションテスト."""

    def test_epochs_missing_failure(self, validator, mocker):
        """epochs未設定でバリデーション失敗."""
        mock_logger = mocker.Mock()
        config = {"batch_size": 32, "model_name": "resnet18"}

        result = validator.validate(config, mock_logger)

        assert result is False
        mock_logger.error.assert_called_with(
            "epochs が設定されていません。configs/pochi_config.py で "
            "エポック数を設定してください。"
        )

    def test_epochs_invalid_type_failure(self, validator, mocker):
        """epochsが整数でない場合バリデーション失敗."""
        mock_logger = mocker.Mock()

        # 文字列の場合
        config = {"epochs": "50", "batch_size": 32, "model_name": "resnet18"}
        result = validator.validate(config, mock_logger)
        assert result is False
        mock_logger.error.assert_called_with(
            "epochs は整数である必要があります。現在の型: str, 現在の値: 50"
        )

        # 浮動小数点数の場合
        mock_logger.reset_mock()
        config = {"epochs": 50.0, "batch_size": 32, "model_name": "resnet18"}
        result = validator.validate(config, mock_logger)
        assert result is False
        mock_logger.error.assert_called_with(
            "epochs は整数である必要があります。現在の型: float, 現在の値: 50.0"
        )

        # ブール値の場合
        mock_logger.reset_mock()
        config = {"epochs": True, "batch_size": 32, "model_name": "resnet18"}
        result = validator.validate(config, mock_logger)
        assert result is False
        mock_logger.error.assert_called_with(
            "epochs は整数である必要があります。現在の型: bool, 現在の値: True"
        )

    def test_epochs_non_positive_failure(self, validator, mocker):
        """epochsが0以下の場合バリデーション失敗."""
        mock_logger = mocker.Mock()

        # 0の場合
        config = {"epochs": 0, "batch_size": 32, "model_name": "resnet18"}
        result = validator.validate(config, mock_logger)
        assert result is False
        mock_logger.error.assert_called_with(
            "epochs は正の整数である必要があります。現在の値: 0"
        )

        # 負の数の場合
        mock_logger.reset_mock()
        config = {"epochs": -10, "batch_size": 32, "model_name": "resnet18"}
        result = validator.validate(config, mock_logger)
        assert result is False
        mock_logger.error.assert_called_with(
            "epochs は正の整数である必要があります。現在の値: -10"
        )

    def test_epochs_valid_success(self, validator, mocker):
        """有効なepochsでバリデーション成功."""
        mock_logger = mocker.Mock()
        config = {"epochs": 50, "batch_size": 32, "model_name": "resnet18"}

        result = validator.validate(config, mock_logger)

        assert result is True
        mock_logger.info.assert_any_call("エポック数: 50")


class TestBatchSizeValidation:
    """batch_sizeパラメータのバリデーションテスト."""

    def test_batch_size_missing_failure(self, validator, mocker):
        """batch_size未設定でバリデーション失敗."""
        mock_logger = mocker.Mock()
        config = {"epochs": 50, "model_name": "resnet18"}

        result = validator.validate(config, mock_logger)

        assert result is False
        mock_logger.error.assert_called_with(
            "batch_size が設定されていません。configs/pochi_config.py で "
            "バッチサイズを設定してください。"
        )

    def test_batch_size_invalid_type_failure(self, validator, mocker):
        """batch_sizeが整数でない場合バリデーション失敗."""
        mock_logger = mocker.Mock()

        # 文字列の場合
        config = {"epochs": 50, "batch_size": "32", "model_name": "resnet18"}
        result = validator.validate(config, mock_logger)
        assert result is False
        mock_logger.error.assert_called_with(
            "batch_size は整数である必要があります。現在の型: str, 現在の値: 32"
        )

        # 浮動小数点数の場合
        mock_logger.reset_mock()
        config = {"epochs": 50, "batch_size": 32.0, "model_name": "resnet18"}
        result = validator.validate(config, mock_logger)
        assert result is False
        mock_logger.error.assert_called_with(
            "batch_size は整数である必要があります。現在の型: float, 現在の値: 32.0"
        )

        # ブール値の場合
        mock_logger.reset_mock()
        config = {"epochs": 50, "batch_size": True, "model_name": "resnet18"}
        result = validator.validate(config, mock_logger)
        assert result is False
        mock_logger.error.assert_called_with(
            "batch_size は整数である必要があります。現在の型: bool, 現在の値: True"
        )

    def test_batch_size_non_positive_failure(self, validator, mocker):
        """batch_sizeが0以下の場合バリデーション失敗."""
        mock_logger = mocker.Mock()

        # 0の場合
        config = {"epochs": 50, "batch_size": 0, "model_name": "resnet18"}
        result = validator.validate(config, mock_logger)
        assert result is False
        mock_logger.error.assert_called_with(
            "batch_size は正の整数である必要があります。現在の値: 0"
        )

        # 負の数の場合
        mock_logger.reset_mock()
        config = {"epochs": 50, "batch_size": -8, "model_name": "resnet18"}
        result = validator.validate(config, mock_logger)
        assert result is False
        mock_logger.error.assert_called_with(
            "batch_size は正の整数である必要があります。現在の値: -8"
        )

    def test_batch_size_valid_success(self, validator, mocker):
        """有効なbatch_sizeでバリデーション成功."""
        mock_logger = mocker.Mock()
        config = {"epochs": 50, "batch_size": 32, "model_name": "resnet18"}

        result = validator.validate(config, mock_logger)

        assert result is True
        mock_logger.info.assert_any_call("バッチサイズ: 32")


class TestModelNameValidation:
    """model_nameパラメータのバリデーションテスト."""

    def test_model_name_missing_failure(self, validator, mocker):
        """model_name未設定でバリデーション失敗."""
        mock_logger = mocker.Mock()
        config = {"epochs": 50, "batch_size": 32}

        result = validator.validate(config, mock_logger)

        assert result is False
        mock_logger.error.assert_called_with(
            "model_name が設定されていません。configs/pochi_config.py で "
            "モデル名を設定してください。"
        )

    def test_model_name_invalid_type_failure(self, validator, mocker):
        """model_nameが文字列でない場合バリデーション失敗."""
        mock_logger = mocker.Mock()

        # 整数の場合
        config = {"epochs": 50, "batch_size": 32, "model_name": 18}
        result = validator.validate(config, mock_logger)
        assert result is False
        mock_logger.error.assert_called_with(
            "model_name は文字列である必要があります。現在の型: int, 現在の値: 18"
        )

        # ブール値の場合
        mock_logger.reset_mock()
        config = {"epochs": 50, "batch_size": 32, "model_name": True}
        result = validator.validate(config, mock_logger)
        assert result is False
        mock_logger.error.assert_called_with(
            "model_name は文字列である必要があります。現在の型: bool, 現在の値: True"
        )

    def test_model_name_unsupported_failure(self, validator, mocker):
        """サポートされていないmodel_nameでバリデーション失敗."""
        mock_logger = mocker.Mock()
        config = {"epochs": 50, "batch_size": 32, "model_name": "vgg16"}

        result = validator.validate(config, mock_logger)

        assert result is False
        mock_logger.error.assert_called_with(
            "サポートされていないモデル名です: vgg16. "
            "サポート対象: ['resnet18', 'resnet34', 'resnet50']"
        )

    def test_model_name_valid_success(self, validator, mocker):
        """有効なmodel_nameでバリデーション成功."""
        mock_logger = mocker.Mock()

        # resnet18
        config = {"epochs": 50, "batch_size": 32, "model_name": "resnet18"}
        result = validator.validate(config, mock_logger)
        assert result is True
        mock_logger.info.assert_any_call("モデル名: resnet18")

        # resnet34
        mock_logger.reset_mock()
        config = {"epochs": 50, "batch_size": 32, "model_name": "resnet34"}
        result = validator.validate(config, mock_logger)
        assert result is True
        mock_logger.info.assert_any_call("モデル名: resnet34")

        # resnet50
        mock_logger.reset_mock()
        config = {"epochs": 50, "batch_size": 32, "model_name": "resnet50"}
        result = validator.validate(config, mock_logger)
        assert result is True
        mock_logger.info.assert_any_call("モデル名: resnet50")


class TestTrainingValidatorIntegration:
    """TrainingValidator統合テスト."""

    def test_all_valid_parameters_success(self, validator, mocker):
        """すべてのパラメータが有効な場合バリデーション成功."""
        mock_logger = mocker.Mock()
        config = {"epochs": 100, "batch_size": 64, "model_name": "resnet50"}

        result = validator.validate(config, mock_logger)

        assert result is True
        mock_logger.info.assert_any_call("エポック数: 100")
        mock_logger.info.assert_any_call("バッチサイズ: 64")
        mock_logger.info.assert_any_call("モデル名: resnet50")

    def test_multiple_invalid_parameters_failure(self, validator, mocker):
        """複数のパラメータが無効な場合、最初のエラーでバリデーション失敗."""
        mock_logger = mocker.Mock()
        config = {
            "epochs": "invalid",  # 最初のエラー
            "batch_size": -1,  # このエラーには到達しない
            "model_name": "invalid_model",  # このエラーには到達しない
        }

        result = validator.validate(config, mock_logger)

        assert result is False
        # epochsの型エラーが最初に検出される
        mock_logger.error.assert_called_with(
            "epochs は整数である必要があります。現在の型: str, 現在の値: invalid"
        )

    def test_boundary_values_success(self, validator, mocker):
        """境界値でバリデーション成功."""
        mock_logger = mocker.Mock()
        config = {
            "epochs": 1,  # 最小有効値
            "batch_size": 1,  # 最小有効値
            "model_name": "resnet18",
        }

        result = validator.validate(config, mock_logger)

        assert result is True
        mock_logger.info.assert_any_call("エポック数: 1")
        mock_logger.info.assert_any_call("バッチサイズ: 1")
        mock_logger.info.assert_any_call("モデル名: resnet18")
