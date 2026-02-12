"""
TransformValidatorのテスト.
"""

import logging
from unittest.mock import Mock

import pytest

from pochitrain.validation.validators.transform_validator import TransformValidator


@pytest.fixture
def validator():
    """TransformValidatorのfixture."""
    return TransformValidator()


@pytest.fixture
def mock_logger():
    """テスト用ロガーのfixture."""
    return Mock(spec=logging.Logger)


def test_train_transform_none_validation_fails(validator, mock_logger):
    """train_transform設定がNoneの場合はバリデーションが失敗することをテスト."""
    config = {"train_transform": None, "val_transform": "dummy_transform"}

    result = validator.validate(config, mock_logger)

    # アサーション
    assert result is False
    mock_logger.error.assert_called_once_with(
        "train_transform が必須です。configs/pochi_train_config.py で "
        "transforms.Compose([...]) を train_transform として定義してください。"
    )


def test_val_transform_none_validation_fails(validator, mock_logger):
    """val_transform設定がNoneの場合はバリデーションが失敗することをテスト."""
    config = {"train_transform": "dummy_transform", "val_transform": None}

    result = validator.validate(config, mock_logger)

    # アサーション
    assert result is False
    mock_logger.error.assert_called_once_with(
        "val_transform が必須です。configs/pochi_train_config.py で "
        "transforms.Compose([...]) を val_transform として定義してください。"
    )


def test_both_transforms_none_validation_fails(validator, mock_logger):
    """両方のtransform設定がNoneの場合はバリデーションが失敗することをテスト."""
    config = {"train_transform": None, "val_transform": None}

    result = validator.validate(config, mock_logger)

    # アサーション（train_transformで先に失敗）
    assert result is False
    mock_logger.error.assert_called_once_with(
        "train_transform が必須です。configs/pochi_train_config.py で "
        "transforms.Compose([...]) を train_transform として定義してください。"
    )


def test_transforms_missing_from_config(validator, mock_logger):
    """設定辞書にtransformキーがない場合のテスト."""
    config = {}  # transformキーなし

    result = validator.validate(config, mock_logger)

    # アサーション（train_transform=Noneと同じ扱い）
    assert result is False
    mock_logger.error.assert_called_once_with(
        "train_transform が必須です。configs/pochi_train_config.py で "
        "transforms.Compose([...]) を train_transform として定義してください。"
    )


def test_both_transforms_valid(validator, mock_logger):
    """両方のtransform設定が有効な場合はバリデーションが成功することをテスト."""
    config = {
        "train_transform": "valid_train_transform",
        "val_transform": "valid_val_transform",
    }

    result = validator.validate(config, mock_logger)

    # アサーション
    assert result is True
    mock_logger.error.assert_not_called()
