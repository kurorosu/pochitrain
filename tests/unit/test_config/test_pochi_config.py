"""PochiConfig のテスト."""

from typing import Any

import pytest

from pochitrain.config.pochi_config import PochiConfig


def _build_minimum_config() -> dict[str, Any]:
    """必須キーのみを持つ最小設定を返す.

    Returns:
        PochiConfig.from_dict の入力に使う最小設定.
    """
    return {
        "model_name": "resnet18",
        "num_classes": 2,
        "device": "cpu",
        "epochs": 10,
        "batch_size": 8,
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "train_data_root": "data/train",
        "train_transform": object(),
        "val_transform": object(),
        "enable_layer_wise_lr": False,
    }


class TestPochiConfigFromDict:
    """PochiConfig.from_dict のテスト."""

    def test_missing_required_keys_raises_value_error(self) -> None:
        """必須キー不足時に ValueError になることを確認する."""
        with pytest.raises(ValueError, match="Missing required config keys"):
            PochiConfig.from_dict({})

    def test_optional_defaults_are_applied(self) -> None:
        """任意キー未指定時にデフォルト値が適用されることを確認する."""
        config = PochiConfig.from_dict(_build_minimum_config())

        assert config.pretrained is True
        assert config.cudnn_benchmark is False
        assert config.work_dir == "work_dirs"
        assert config.num_workers == 0
        assert config.mean == [0.485, 0.456, 0.406]
        assert config.std == [0.229, 0.224, 0.225]
        assert config.early_stopping is None
        assert config.confusion_matrix_config is None
        assert config.optuna is None
        assert config.gradient_tracking_config.record_frequency == 1
        assert config.layer_wise_lr_config.layer_rates == {}

    def test_top_level_optuna_keys_enable_optuna_config(self) -> None:
        """optuna dict がなくても top-level キーで optuna が有効になる."""
        payload = _build_minimum_config()
        payload["n_trials"] = 5
        payload["direction"] = "minimize"

        config = PochiConfig.from_dict(payload)

        assert config.optuna is not None
        assert config.optuna.n_trials == 5
        assert config.optuna.direction == "minimize"

    def test_nested_configs_and_optuna_merge(self) -> None:
        """ネスト設定と optuna マージが正しく反映されることを確認する."""
        payload = _build_minimum_config()
        payload.update(
            {
                "early_stopping": {
                    "enabled": True,
                    "patience": 3,
                    "min_delta": 0.01,
                    "monitor": "val_loss",
                },
                "confusion_matrix_config": {
                    "title": "CM",
                    "figsize": (12, 10),
                    "cmap": "Reds",
                },
                "gradient_tracking_config": {
                    "record_frequency": 2,
                    "exclude_patterns": ["layer4\\."],
                    "group_by_block": False,
                    "aggregation_method": "mean",
                },
                "layer_wise_lr_config": {
                    "layer_rates": {"fc": 0.01},
                    "graph_config": {"use_log_scale": False},
                },
                "optuna": {
                    "n_trials": 2,
                    "study_name": "nested_name",
                    "search_space": {"lr": {"type": "float"}},
                },
                "n_trials": 9,
                "direction": "minimize",
            }
        )

        config = PochiConfig.from_dict(payload)

        assert config.early_stopping is not None
        assert config.early_stopping.enabled is True
        assert config.early_stopping.patience == 3

        assert config.confusion_matrix_config is not None
        assert config.confusion_matrix_config.title == "CM"
        assert config.confusion_matrix_config.figsize == (12, 10)
        assert config.confusion_matrix_config.cmap == "Reds"

        assert config.gradient_tracking_config.record_frequency == 2
        assert config.gradient_tracking_config.exclude_patterns == ["layer4\\."]
        assert config.gradient_tracking_config.group_by_block is False
        assert config.gradient_tracking_config.aggregation_method == "mean"

        assert config.layer_wise_lr_config.layer_rates["fc"] == 0.01
        assert config.layer_wise_lr_config.graph_config.use_log_scale is False

        assert config.optuna is not None
        assert config.optuna.n_trials == 9
        assert config.optuna.study_name == "nested_name"
        assert config.optuna.direction == "minimize"
        assert "lr" in config.optuna.search_space


class TestPochiConfigToDict:
    """PochiConfig.to_dict のテスト."""

    def test_to_dict_flattens_optuna_and_nested_dataclasses(self) -> None:
        """to_dict で optuna とネスト dataclass が期待形になることを確認する."""
        payload = _build_minimum_config()
        payload["layer_wise_lr_config"] = {"layer_rates": {"fc": 0.02}}
        payload["optuna"] = {"n_trials": 7, "study_name": "study_x"}

        config = PochiConfig.from_dict(payload)
        result = config.to_dict()

        assert "optuna" not in result
        assert result["n_trials"] == 7
        assert result["study_name"] == "study_x"
        assert isinstance(result["layer_wise_lr_config"], dict)
        assert result["layer_wise_lr_config"]["layer_rates"]["fc"] == 0.02
        assert isinstance(result["gradient_tracking_config"], dict)
