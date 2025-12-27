"""OptimizerValidatorのテスト."""

import pytest

from pochitrain.validation.validators.optimizer_validator import OptimizerValidator


@pytest.fixture
def validator():
    """OptimizerValidatorのfixture."""
    return OptimizerValidator()


def test_learning_rate_missing_failure(validator, mocker):
    """learning_rate未設定でバリデーション失敗."""
    mock_logger = mocker.Mock()
    config = {"optimizer": "Adam"}

    result = validator.validate(config, mock_logger)

    assert result is False
    mock_logger.error.assert_called_with(
        "learning_rate が設定されていません。configs/pochi_train_config.py で "
        "学習率を設定してください。"
    )


def test_learning_rate_invalid_type_failure(validator, mocker):
    """learning_rateが数値でない場合バリデーション失敗."""
    mock_logger = mocker.Mock()
    config = {"learning_rate": "0.001", "optimizer": "Adam"}

    result = validator.validate(config, mock_logger)

    assert result is False
    mock_logger.error.assert_called_with(
        "learning_rate は数値である必要があります。現在の型: <class 'str'>"
    )


def test_learning_rate_out_of_range_failure(validator, mocker):
    """learning_rateが範囲外の場合バリデーション失敗."""
    mock_logger = mocker.Mock()

    # 0以下のケース
    config = {"learning_rate": 0, "optimizer": "Adam"}
    result = validator.validate(config, mock_logger)
    assert result is False

    # 1.0超過のケース
    config = {"learning_rate": 1.5, "optimizer": "Adam"}
    result = validator.validate(config, mock_logger)
    assert result is False


def test_optimizer_missing_failure(validator, mocker):
    """optimizer未設定でバリデーション失敗."""
    mock_logger = mocker.Mock()
    config = {"learning_rate": 0.001}

    result = validator.validate(config, mock_logger)

    assert result is False
    mock_logger.error.assert_called_with(
        "optimizer が設定されていません。configs/pochi_train_config.py で "
        "最適化器を設定してください。"
    )


def test_optimizer_invalid_type_failure(validator, mocker):
    """optimizerが文字列でない場合バリデーション失敗."""
    mock_logger = mocker.Mock()
    config = {"learning_rate": 0.001, "optimizer": 123}

    result = validator.validate(config, mock_logger)

    assert result is False
    mock_logger.error.assert_called_with(
        "optimizer は文字列である必要があります。現在の型: <class 'int'>"
    )


def test_optimizer_unsupported_failure(validator, mocker):
    """サポートされていないoptimizer名でバリデーション失敗."""
    mock_logger = mocker.Mock()
    config = {"learning_rate": 0.001, "optimizer": "RMSprop"}

    result = validator.validate(config, mock_logger)

    assert result is False
    mock_logger.error.assert_called_with(
        "サポートされていない最適化器です: RMSprop. "
        "サポート対象: ['Adam', 'AdamW', 'SGD']"
    )


def test_valid_adam_success(validator, mocker):
    """有効なAdam設定でバリデーション成功."""
    mock_logger = mocker.Mock()
    config = {"learning_rate": 0.001, "optimizer": "Adam"}

    result = validator.validate(config, mock_logger)

    assert result is True
    mock_logger.info.assert_any_call("学習率: 0.001")
    mock_logger.info.assert_any_call("最適化器: Adam")


def test_valid_sgd_success(validator, mocker):
    """有効なSGD設定でバリデーション成功."""
    mock_logger = mocker.Mock()
    config = {"learning_rate": 0.1, "optimizer": "SGD"}

    result = validator.validate(config, mock_logger)

    assert result is True
    mock_logger.info.assert_any_call("学習率: 0.1")
    mock_logger.info.assert_any_call("最適化器: SGD")


def test_learning_rate_boundary_values(validator, mocker):
    """learning_rateの境界値テスト."""
    mock_logger = mocker.Mock()

    # 最小値（0に近い値）
    config = {"learning_rate": 0.0001, "optimizer": "Adam"}
    result = validator.validate(config, mock_logger)
    assert result is True

    # 最大値（1.0）
    config = {"learning_rate": 1.0, "optimizer": "Adam"}
    result = validator.validate(config, mock_logger)
    assert result is True
