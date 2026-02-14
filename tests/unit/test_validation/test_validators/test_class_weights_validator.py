"""ClassWeightsValidatorのテスト.

`num_classes` と `class_weights` の失敗系/正常系を検証する.
"""

import logging
from unittest.mock import Mock

import pytest

from pochitrain.validation.validators.class_weights_validator import (
    ClassWeightsValidator,
)
from tests.unit.test_validation.conftest import assert_error_called_with_substring


@pytest.fixture
def validator() -> ClassWeightsValidator:
    """ClassWeightsValidatorのfixture."""

    return ClassWeightsValidator()


@pytest.fixture
def mock_logger() -> Mock:
    """検証用ロガーのfixture."""

    return Mock(spec=logging.Logger)


@pytest.mark.parametrize(
    "config,error_key",
    [
        ({"class_weights": [1.0, 2.0]}, "num_classes"),
        ({"num_classes": None, "class_weights": [1.0, 2.0]}, "num_classes"),
        ({"num_classes": "4", "class_weights": [1.0, 2.0]}, "num_classes"),
        ({"num_classes": 0, "class_weights": [1.0, 2.0]}, "num_classes"),
        ({"num_classes": -1, "class_weights": [1.0, 2.0]}, "num_classes"),
        ({"num_classes": 4, "class_weights": "invalid"}, "class_weights"),
        ({"num_classes": 4, "class_weights": [1.0, 2.0]}, "num_classes"),
        ({"num_classes": 3, "class_weights": [1.0, "bad", 3.0]}, "class_weights[1]"),
        ({"num_classes": 3, "class_weights": [1.0, -2.0, 3.0]}, "class_weights[1]"),
        ({"num_classes": 3, "class_weights": [1.0, 0.0, 3.0]}, "class_weights[1]"),
    ],
)
def test_validate_failure_cases(
    validator: ClassWeightsValidator,
    mock_logger: Mock,
    config: dict,
    error_key: str,
) -> None:
    """代表的な失敗ケースでFalseとエラーログを確認."""

    assert validator.validate(config, mock_logger) is False
    assert_error_called_with_substring(mock_logger, error_key)


def test_class_weights_none_is_valid(
    validator: ClassWeightsValidator,
    mock_logger: Mock,
) -> None:
    """`class_weights=None` は有効として通ることを確認."""

    config = {"num_classes": 4, "class_weights": None}

    assert validator.validate(config, mock_logger) is True
    mock_logger.error.assert_not_called()


@pytest.mark.parametrize(
    "weights",
    [
        [1.0, 2.5, 0.8],
        (1.0, 2.0, 1.5, 3.0),
        [1, 3],
        [1, 2.5, 3],
    ],
)
def test_valid_weight_patterns(
    validator: ClassWeightsValidator,
    mock_logger: Mock,
    weights: list,
) -> None:
    """有効な重みパターンでエラーログなしに通過することを確認."""

    config = {"num_classes": len(weights), "class_weights": weights}

    assert validator.validate(config, mock_logger) is True
    mock_logger.error.assert_not_called()
