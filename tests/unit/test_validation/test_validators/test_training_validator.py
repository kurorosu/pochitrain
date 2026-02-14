"""TrainingValidatorのテスト.

必須項目, 型/値検証, 正常系, 先頭エラー停止を確認する.
"""

import logging
from unittest.mock import Mock

import pytest

from pochitrain.validation.validators.training_validator import TrainingValidator
from tests.unit.test_validation.conftest import (
    assert_error_called_with_substring,
    assert_info_or_debug_called_with,
)


@pytest.fixture
def validator() -> TrainingValidator:
    """TrainingValidatorのfixture."""

    return TrainingValidator()


@pytest.fixture
def mock_logger() -> Mock:
    """検証用ロガーのfixture."""

    return Mock(spec=logging.Logger)


@pytest.mark.parametrize(
    "config,error_key",
    [
        ({"batch_size": 32, "model_name": "resnet18"}, "epochs"),
        ({"epochs": "10", "batch_size": 32, "model_name": "resnet18"}, "epochs"),
        ({"epochs": True, "batch_size": 32, "model_name": "resnet18"}, "epochs"),
        ({"epochs": 0, "batch_size": 32, "model_name": "resnet18"}, "epochs"),
        ({"epochs": 10, "model_name": "resnet18"}, "batch_size"),
        ({"epochs": 10, "batch_size": "32", "model_name": "resnet18"}, "batch_size"),
        ({"epochs": 10, "batch_size": -1, "model_name": "resnet18"}, "batch_size"),
        ({"epochs": 10, "batch_size": 32}, "model_name"),
        ({"epochs": 10, "batch_size": 32, "model_name": 18}, "model_name"),
        ({"epochs": 10, "batch_size": 32, "model_name": "vgg16"}, "vgg16"),
    ],
)
def test_validate_failure_cases(
    validator: TrainingValidator,
    mock_logger: Mock,
    config: dict,
    error_key: str,
) -> None:
    """代表的な失敗パターンでFalseとエラーログを確認."""

    assert validator.validate(config, mock_logger) is False
    assert_error_called_with_substring(mock_logger, error_key)


def test_validate_success(validator: TrainingValidator, mock_logger: Mock) -> None:
    """有効設定でTrueになり, 主要値がログ出力されることを確認."""

    config = {"epochs": 50, "batch_size": 32, "model_name": "resnet18"}

    assert validator.validate(config, mock_logger) is True
    assert_info_or_debug_called_with(mock_logger, "50")
    assert_info_or_debug_called_with(mock_logger, "32")
    assert_info_or_debug_called_with(mock_logger, "resnet18")


def test_validate_stops_at_first_error(
    validator: TrainingValidator,
    mock_logger: Mock,
) -> None:
    """複数不正時に最初のエラーで停止することを確認."""

    config = {
        "epochs": "invalid",
        "batch_size": -1,
        "model_name": "invalid_model",
    }

    assert validator.validate(config, mock_logger) is False
    assert_error_called_with_substring(mock_logger, "epochs")
