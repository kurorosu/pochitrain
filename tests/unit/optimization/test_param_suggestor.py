"""ParamSuggestorのユニットテスト."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from pochitrain.optimization.param_suggestor import (
    DefaultParamSuggestor,
    LayerWiseLRSuggestor,
)


class TestDefaultParamSuggestor:
    """DefaultParamSuggestorのテスト."""

    def test_suggest_float_parameter(self) -> None:
        """float型パラメータのサジェストをテスト."""
        search_space = {
            "learning_rate": {
                "type": "float",
                "low": 1e-5,
                "high": 1e-1,
                "log": True,
            },
        }
        suggestor = DefaultParamSuggestor(search_space)

        # Optunaのtrialをモック
        trial = MagicMock()
        trial.suggest_float.return_value = 0.001

        params = suggestor.suggest(trial)

        assert "learning_rate" in params
        assert params["learning_rate"] == 0.001
        trial.suggest_float.assert_called_once_with(
            "learning_rate", 1e-5, 1e-1, log=True
        )

    def test_suggest_int_parameter(self) -> None:
        """int型パラメータのサジェストをテスト."""
        search_space = {
            "batch_size": {
                "type": "int",
                "low": 8,
                "high": 64,
            },
        }
        suggestor = DefaultParamSuggestor(search_space)

        trial = MagicMock()
        trial.suggest_int.return_value = 32

        params = suggestor.suggest(trial)

        assert "batch_size" in params
        assert params["batch_size"] == 32
        trial.suggest_int.assert_called_once_with("batch_size", 8, 64, log=False)

    def test_suggest_categorical_parameter(self) -> None:
        """categorical型パラメータのサジェストをテスト."""
        search_space = {
            "optimizer": {
                "type": "categorical",
                "choices": ["SGD", "Adam", "AdamW"],
            },
        }
        suggestor = DefaultParamSuggestor(search_space)

        trial = MagicMock()
        trial.suggest_categorical.return_value = "Adam"

        params = suggestor.suggest(trial)

        assert "optimizer" in params
        assert params["optimizer"] == "Adam"
        trial.suggest_categorical.assert_called_once_with(
            "optimizer", ["SGD", "Adam", "AdamW"]
        )

    def test_suggest_multiple_parameters(self) -> None:
        """複数パラメータのサジェストをテスト."""
        search_space: dict[str, dict[str, Any]] = {
            "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-1, "log": True},
            "batch_size": {"type": "categorical", "choices": [16, 32, 64]},
            "optimizer": {"type": "categorical", "choices": ["SGD", "Adam"]},
        }
        suggestor = DefaultParamSuggestor(search_space)

        trial = MagicMock()
        trial.suggest_float.return_value = 0.001
        trial.suggest_categorical.side_effect = [32, "Adam"]

        params = suggestor.suggest(trial)

        assert len(params) == 3
        assert "learning_rate" in params
        assert "batch_size" in params
        assert "optimizer" in params

    def test_unknown_parameter_type_raises_error(self) -> None:
        """未知のパラメータタイプでエラーが発生することをテスト."""
        search_space = {
            "unknown_param": {
                "type": "unknown",
                "value": 42,
            },
        }
        suggestor = DefaultParamSuggestor(search_space)
        trial = MagicMock()

        with pytest.raises(ValueError, match="Unknown parameter type"):
            suggestor.suggest(trial)


class TestLayerWiseLRSuggestor:
    """LayerWiseLRSuggestorのテスト."""

    def test_suggest_layer_wise_lr(self) -> None:
        """層別学習率のサジェストをテスト."""
        layers = ["conv1", "layer1", "fc"]
        suggestor = LayerWiseLRSuggestor(
            layers=layers,
            base_lr_range=(1e-5, 1e-2),
            layer_lr_scale_range=(0.1, 10.0),
        )

        trial = MagicMock()
        trial.suggest_float.side_effect = [
            0.001,  # base_learning_rate
            1.0,  # lr_scale_conv1
            2.0,  # lr_scale_layer1
            5.0,  # lr_scale_fc
        ]

        params = suggestor.suggest(trial)

        assert params["learning_rate"] == 0.001
        assert params["enable_layer_wise_lr"] is True
        assert "layer_wise_lr_config" in params
        assert "layer_rates" in params["layer_wise_lr_config"]

        layer_rates = params["layer_wise_lr_config"]["layer_rates"]
        assert layer_rates["conv1"] == 0.001 * 1.0
        assert layer_rates["layer1"] == 0.001 * 2.0
        assert layer_rates["fc"] == 0.001 * 5.0
