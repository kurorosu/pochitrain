"""ParamSuggestor のユニットテスト."""

from typing import Any

import optuna
import pytest

from pochitrain.optimization.param_suggestor import (
    DefaultParamSuggestor,
    LayerWiseLRSuggestor,
)


class TestDefaultParamSuggestor:
    """DefaultParamSuggestor のテスト."""

    def test_suggest_float_parameter(self) -> None:
        """float パラメータを提案できることを検証する."""
        search_space = {
            "learning_rate": {
                "type": "float",
                "low": 1e-5,
                "high": 1e-1,
                "log": True,
            },
        }
        suggestor = DefaultParamSuggestor(search_space)

        trial = optuna.trial.FixedTrial({"learning_rate": 0.001})
        params = suggestor.suggest(trial)

        assert params["learning_rate"] == pytest.approx(0.001)

    def test_suggest_int_parameter(self) -> None:
        """int パラメータを提案できることを検証する."""
        search_space = {
            "batch_size": {
                "type": "int",
                "low": 8,
                "high": 64,
            },
        }
        suggestor = DefaultParamSuggestor(search_space)

        trial = optuna.trial.FixedTrial({"batch_size": 32})
        params = suggestor.suggest(trial)

        assert params["batch_size"] == 32

    def test_suggest_categorical_parameter(self) -> None:
        """categorical パラメータを提案できることを検証する."""
        search_space = {
            "optimizer": {
                "type": "categorical",
                "choices": ["SGD", "Adam", "AdamW"],
            },
        }
        suggestor = DefaultParamSuggestor(search_space)

        trial = optuna.trial.FixedTrial({"optimizer": "Adam"})
        params = suggestor.suggest(trial)

        assert params["optimizer"] == "Adam"

    def test_suggest_multiple_parameters(self) -> None:
        """複数パラメータを同時に提案できることを検証する."""
        search_space: dict[str, dict[str, Any]] = {
            "learning_rate": {
                "type": "float",
                "low": 1e-5,
                "high": 1e-1,
                "log": True,
            },
            "batch_size": {"type": "categorical", "choices": [16, 32, 64]},
            "optimizer": {"type": "categorical", "choices": ["SGD", "Adam"]},
        }
        suggestor = DefaultParamSuggestor(search_space)

        trial = optuna.trial.FixedTrial(
            {
                "learning_rate": 0.001,
                "batch_size": 32,
                "optimizer": "Adam",
            }
        )
        params = suggestor.suggest(trial)

        assert params == {
            "learning_rate": 0.001,
            "batch_size": 32,
            "optimizer": "Adam",
        }

    def test_unknown_parameter_type_raises_error(self) -> None:
        """未知のパラメータ型では例外になることを検証する."""
        search_space = {
            "unknown_param": {
                "type": "unknown",
                "value": 42,
            },
        }
        suggestor = DefaultParamSuggestor(search_space)

        with pytest.raises(ValueError, match="Unknown parameter type"):
            suggestor.suggest(optuna.trial.FixedTrial({}))


class TestLayerWiseLRSuggestor:
    """LayerWiseLRSuggestor のテスト."""

    def test_suggest_layer_wise_lr(self) -> None:
        """層ごとの学習率提案を返すことを検証する."""
        layers = ["conv1", "layer1", "fc"]
        suggestor = LayerWiseLRSuggestor(
            layers=layers,
            base_lr_range=(1e-5, 1e-2),
            layer_lr_scale_range=(0.1, 10.0),
        )

        trial = optuna.trial.FixedTrial(
            {
                "base_learning_rate": 0.001,
                "lr_scale_conv1": 1.0,
                "lr_scale_layer1": 2.0,
                "lr_scale_fc": 5.0,
            }
        )
        params = suggestor.suggest(trial)

        assert params["learning_rate"] == pytest.approx(0.001)
        assert params["enable_layer_wise_lr"] is True

        layer_rates = params["layer_wise_lr_config"]["layer_rates"]
        assert layer_rates["conv1"] == pytest.approx(0.001)
        assert layer_rates["layer1"] == pytest.approx(0.002)
        assert layer_rates["fc"] == pytest.approx(0.005)
