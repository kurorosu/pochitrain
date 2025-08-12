"""
DeviceValidatorのテスト.
"""

import pytest

from pochitrain.validation.validators.device_validator import DeviceValidator


@pytest.fixture
def validator():
    """DeviceValidatorのfixture."""
    return DeviceValidator()


def test_device_none_validation_fails(validator, mocker):
    """device設定がNoneの場合はバリデーションが失敗することをテスト."""
    mock_logger = mocker.Mock()
    config = {"device": None}

    result = validator.validate(config, mock_logger)

    # アサーション
    assert result is False
    # エラーメッセージが出力されることを確認
    assert mock_logger.error.call_count == 2
    mock_logger.error.assert_any_call(
        "device設定が必須です。configs/pochi_config.pyでdeviceを'cuda'または'cpu'に設定してください。"
    )
    mock_logger.error.assert_any_call("例: device = 'cuda' または device = 'cpu'")


def test_device_cpu_shows_warning(validator, mocker):
    """device設定が'cpu'の場合は警告メッセージを表示することをテスト."""
    mock_logger = mocker.Mock()
    config = {"device": "cpu"}

    result = validator.validate(config, mock_logger)

    # アサーション
    assert result is True
    # 警告メッセージが出力されることを確認
    assert mock_logger.warning.call_count == 3
    mock_logger.warning.assert_any_call("⚠️  CPU使用モードで実行中です")
    mock_logger.warning.assert_any_call(
        "⚠️  GPU使用を推奨します（大幅な性能向上が期待できます）"
    )
    mock_logger.warning.assert_any_call(
        "⚠️  GPU使用時: device = 'cuda' に設定してください"
    )


def test_device_cuda_no_warning(validator, mocker):
    """device設定が'cuda'の場合は警告メッセージを表示しないことをテスト."""
    mock_logger = mocker.Mock()
    config = {"device": "cuda"}

    result = validator.validate(config, mock_logger)

    # アサーション
    assert result is True
    # 警告メッセージが出力されないことを確認
    mock_logger.warning.assert_not_called()


def test_device_missing_from_config(validator, mocker):
    """設定辞書にdeviceキーがない場合のテスト."""
    mock_logger = mocker.Mock()
    config = {}  # deviceキーなし

    result = validator.validate(config, mock_logger)

    # アサーション（device=Noneと同じ扱い）
    assert result is False
    assert mock_logger.error.call_count == 2
