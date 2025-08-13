"""
TransformValidatorのテスト.
"""

import pytest

from pochitrain.validation.validators.transform_validator import TransformValidator


@pytest.fixture
def validator():
    """TransformValidatorのfixture."""
    return TransformValidator()


def test_train_transform_none_validation_fails(validator, mocker):
    """train_transform設定がNoneの場合はバリデーションが失敗することをテスト."""
    mock_logger = mocker.Mock()
    config = {"train_transform": None, "val_transform": "dummy_transform"}

    result = validator.validate(config, mock_logger)

    # アサーション
    assert result is False
    mock_logger.error.assert_called_once_with(
        "train_transform が必須です。configs/pochi_train_config.py で "
        "transforms.Compose([...]) を train_transform として定義してください。"
    )


def test_val_transform_none_validation_fails(validator, mocker):
    """val_transform設定がNoneの場合はバリデーションが失敗することをテスト."""
    mock_logger = mocker.Mock()
    config = {"train_transform": "dummy_transform", "val_transform": None}

    result = validator.validate(config, mock_logger)

    # アサーション
    assert result is False
    mock_logger.error.assert_called_once_with(
        "val_transform が必須です。configs/pochi_train_config.py で "
        "transforms.Compose([...]) を val_transform として定義してください。"
    )


def test_both_transforms_none_validation_fails(validator, mocker):
    """両方のtransform設定がNoneの場合はバリデーションが失敗することをテスト."""
    mock_logger = mocker.Mock()
    config = {"train_transform": None, "val_transform": None}

    result = validator.validate(config, mock_logger)

    # アサーション（train_transformで先に失敗）
    assert result is False
    mock_logger.error.assert_called_once_with(
        "train_transform が必須です。configs/pochi_train_config.py で "
        "transforms.Compose([...]) を train_transform として定義してください。"
    )


def test_transforms_missing_from_config(validator, mocker):
    """設定辞書にtransformキーがない場合のテスト."""
    mock_logger = mocker.Mock()
    config = {}  # transformキーなし

    result = validator.validate(config, mock_logger)

    # アサーション（train_transform=Noneと同じ扱い）
    assert result is False
    mock_logger.error.assert_called_once_with(
        "train_transform が必須です。configs/pochi_train_config.py で "
        "transforms.Compose([...]) を train_transform として定義してください。"
    )


def test_both_transforms_valid(validator, mocker):
    """両方のtransform設定が有効な場合はバリデーションが成功することをテスト."""
    mock_logger = mocker.Mock()
    config = {
        "train_transform": "valid_train_transform",
        "val_transform": "valid_val_transform",
    }

    result = validator.validate(config, mock_logger)

    # アサーション
    assert result is True
    mock_logger.error.assert_not_called()
