"""ハイパーパラメータサジェスター実装（SRP: 単一責任原則）."""

from typing import Any

import optuna

from pochitrain.optimization.interfaces import IParamSuggestor


class DefaultParamSuggestor(IParamSuggestor):
    """デフォルトのハイパーパラメータサジェスター.

    探索空間の設定に基づいてパラメータを提案する。
    """

    def __init__(self, search_space: dict[str, dict[str, Any]]) -> None:
        """初期化.

        Args:
            search_space: 探索空間の定義
                例:
                {
                    "learning_rate": {
                        "type": "float",
                        "low": 1e-5,
                        "high": 1e-1,
                        "log": True,
                    },
                    "batch_size": {
                        "type": "categorical",
                        "choices": [16, 32, 64],
                    },
                }
        """
        self._search_space = search_space

    def suggest(self, trial: optuna.Trial) -> dict[str, Any]:
        """探索空間に基づいてパラメータを提案する.

        Args:
            trial: Optuna trial オブジェクト

        Returns:
            サジェストされたハイパーパラメータの辞書
        """
        params: dict[str, Any] = {}

        for param_name, config in self._search_space.items():
            param_type = config.get("type", "float")

            if param_type == "float":
                params[param_name] = trial.suggest_float(
                    param_name,
                    config["low"],
                    config["high"],
                    log=config.get("log", False),
                )
            elif param_type == "int":
                params[param_name] = trial.suggest_int(
                    param_name,
                    config["low"],
                    config["high"],
                    log=config.get("log", False),
                )
            elif param_type == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    config["choices"],
                )
            else:
                msg = f"Unknown parameter type: {param_type}"
                raise ValueError(msg)

        return params


class LayerWiseLRSuggestor(IParamSuggestor):
    """層別学習率用のパラメータサジェスター.

    層ごとの学習率を個別に提案する。
    """

    def __init__(
        self,
        layers: list[str],
        base_lr_range: tuple[float, float] = (1e-5, 1e-2),
        layer_lr_scale_range: tuple[float, float] = (0.1, 10.0),
    ) -> None:
        """初期化.

        Args:
            layers: 層名のリスト
            base_lr_range: ベース学習率の範囲
            layer_lr_scale_range: 層ごとのスケール範囲
        """
        self._layers = layers
        self._base_lr_range = base_lr_range
        self._layer_lr_scale_range = layer_lr_scale_range

    def suggest(self, trial: optuna.Trial) -> dict[str, Any]:
        """層別学習率パラメータを提案する.

        Args:
            trial: Optuna trial オブジェクト

        Returns:
            層別学習率設定を含む辞書
        """
        base_lr = trial.suggest_float(
            "base_learning_rate",
            self._base_lr_range[0],
            self._base_lr_range[1],
            log=True,
        )

        layer_rates: dict[str, float] = {}
        for layer in self._layers:
            scale = trial.suggest_float(
                f"lr_scale_{layer}",
                self._layer_lr_scale_range[0],
                self._layer_lr_scale_range[1],
                log=True,
            )
            layer_rates[layer] = base_lr * scale

        return {
            "learning_rate": base_lr,
            "enable_layer_wise_lr": True,
            "layer_wise_lr_config": {
                "layer_rates": layer_rates,
            },
        }
