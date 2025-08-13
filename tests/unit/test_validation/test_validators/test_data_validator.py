"""
DataValidatorのテスト.
"""

import tempfile
from pathlib import Path

import pytest

from pochitrain.validation.validators.data_validator import DataValidator


@pytest.fixture
def validator():
    """DataValidatorのfixture."""
    return DataValidator()


@pytest.fixture
def temp_paths():
    """一時ディレクトリのfixture."""
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)

    # 有効なデータパスを作成
    valid_train_path = temp_path / "train"
    valid_val_path = temp_path / "val"
    valid_train_path.mkdir()
    valid_val_path.mkdir()

    return {
        "temp_path": temp_path,
        "valid_train_path": valid_train_path,
        "valid_val_path": valid_val_path,
    }


def test_train_data_root_none_validation_fails(validator, temp_paths, mocker):
    """train_data_root設定がNoneの場合はバリデーションが失敗することをテスト."""
    mock_logger = mocker.Mock()
    config = {
        "train_data_root": None,
        "val_data_root": str(temp_paths["valid_val_path"]),
    }

    result = validator.validate(config, mock_logger)

    # アサーション
    assert result is False
    mock_logger.error.assert_called_once_with(
        "train_data_root が必須です。configs/pochi_train_config.py で "
        "有効な訓練データパスを設定してください。"
    )


def test_train_data_root_missing_validation_fails(validator, temp_paths, mocker):
    """train_data_root設定が設定辞書にない場合はバリデーションが失敗することをテスト."""
    mock_logger = mocker.Mock()
    config = {"val_data_root": str(temp_paths["valid_val_path"])}

    result = validator.validate(config, mock_logger)

    # アサーション
    assert result is False
    mock_logger.error.assert_called_once_with(
        "train_data_root が必須です。configs/pochi_train_config.py で "
        "有効な訓練データパスを設定してください。"
    )


def test_train_data_root_not_exists_validation_fails(validator, temp_paths, mocker):
    """train_data_rootが存在しない場合はバリデーションが失敗することをテスト."""
    mock_logger = mocker.Mock()
    nonexistent_path = str(temp_paths["temp_path"] / "nonexistent_train")
    config = {
        "train_data_root": nonexistent_path,
        "val_data_root": str(temp_paths["valid_val_path"]),
    }

    result = validator.validate(config, mock_logger)

    # アサーション
    assert result is False
    mock_logger.error.assert_called_once_with(
        f"訓練データパスが存在しません: {nonexistent_path}"
    )


def test_val_data_root_none_validation_fails(validator, temp_paths, mocker):
    """val_data_root設定がNoneの場合はバリデーションが失敗することをテスト."""
    mock_logger = mocker.Mock()
    config = {
        "train_data_root": str(temp_paths["valid_train_path"]),
        "val_data_root": None,
    }

    result = validator.validate(config, mock_logger)

    # アサーション
    assert result is False
    mock_logger.error.assert_called_once_with(
        "val_data_root が必須です。configs/pochi_train_config.py で "
        "有効な検証データパスを設定してください。"
    )


def test_val_data_root_not_exists_validation_fails(validator, temp_paths, mocker):
    """val_data_rootが存在しない場合はバリデーションが失敗することをテスト."""
    mock_logger = mocker.Mock()
    nonexistent_path = str(temp_paths["temp_path"] / "nonexistent_val")
    config = {
        "train_data_root": str(temp_paths["valid_train_path"]),
        "val_data_root": nonexistent_path,
    }

    result = validator.validate(config, mock_logger)

    # アサーション
    assert result is False
    mock_logger.error.assert_called_once_with(
        f"検証データパスが存在しません: {nonexistent_path}"
    )


def test_both_data_paths_valid(validator, temp_paths, mocker):
    """両方のデータパス設定が有効な場合はバリデーションが成功することをテスト."""
    mock_logger = mocker.Mock()
    config = {
        "train_data_root": str(temp_paths["valid_train_path"]),
        "val_data_root": str(temp_paths["valid_val_path"]),
    }

    result = validator.validate(config, mock_logger)

    # アサーション
    assert result is True
    mock_logger.error.assert_not_called()


def test_empty_string_paths_validation_fails(validator, mocker):
    """空文字列のパス設定でバリデーションが失敗することをテスト."""
    mock_logger = mocker.Mock()
    config = {"train_data_root": "", "val_data_root": ""}

    result = validator.validate(config, mock_logger)

    # アサーション（train_data_rootで先に失敗）
    assert result is False
    mock_logger.error.assert_called_once_with(
        "train_data_root が必須です。configs/pochi_train_config.py で "
        "有効な訓練データパスを設定してください。"
    )
