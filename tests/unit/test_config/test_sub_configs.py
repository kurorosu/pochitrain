"""sub_configs のテスト."""

import pytest
from pydantic import ValidationError

from pochitrain.config.sub_configs import (
    ConfusionMatrixConfig,
    EarlyStoppingConfig,
    GradientTrackingConfig,
    LayerWiseLRConfig,
    OptunaConfig,
)


class TestEarlyStoppingConfig:
    """EarlyStoppingConfig のテスト."""

    def test_default_values(self) -> None:
        """デフォルト値が正しく設定されることを確認する."""
        config = EarlyStoppingConfig()

        assert config.enabled is False
        assert config.patience == 10
        assert config.min_delta == 0.0
        assert config.monitor == "val_accuracy"

    def test_model_validate_none_uses_defaults(self) -> None:
        """空 dict 入力でデフォルト値が使われることを確認する."""
        config = EarlyStoppingConfig.model_validate({})

        assert config.enabled is False
        assert config.patience == 10
        assert config.min_delta == 0.0
        assert config.monitor == "val_accuracy"

    def test_invalid_patience_raises_error(self) -> None:
        """patience が 0 以下の場合にエラーになることを確認する."""
        with pytest.raises(ValidationError):
            EarlyStoppingConfig(enabled=False, patience=0)

    def test_negative_min_delta_raises_error(self) -> None:
        """min_delta が負の場合にエラーになることを確認する."""
        with pytest.raises(ValidationError):
            EarlyStoppingConfig(enabled=False, min_delta=-0.1)

    def test_invalid_monitor_raises_error(self) -> None:
        """monitor が不正値の場合にエラーになることを確認する."""
        with pytest.raises(ValidationError):
            EarlyStoppingConfig(enabled=False, monitor="invalid")


class TestLayerWiseLRConfig:
    """LayerWiseLRConfig のテスト."""

    def test_model_validate_builds_nested_graph_config(self) -> None:
        """graph_config を含む辞書を正しく変換できることを確認する."""
        config = LayerWiseLRConfig.model_validate(
            {
                "layer_rates": {"layer4": 0.001, "fc": 0.01},
                "graph_config": {"use_log_scale": False},
            }
        )

        assert config.layer_rates["layer4"] == 0.001
        assert config.layer_rates["fc"] == 0.01
        assert config.graph_config.use_log_scale is False


class TestGradientTrackingConfig:
    """GradientTrackingConfig のテスト."""

    def test_model_validate_overrides_defaults(self) -> None:
        """カスタム設定でデフォルトが上書きされることを確認する."""
        config = GradientTrackingConfig.model_validate(
            {
                "record_frequency": 3,
                "exclude_patterns": ["bn\\."],
                "group_by_block": False,
                "aggregation_method": "mean",
            }
        )

        assert config.record_frequency == 3
        assert config.exclude_patterns == ["bn\\."]
        assert config.group_by_block is False
        assert config.aggregation_method == "mean"

    def test_invalid_record_frequency_raises_error(self) -> None:
        """record_frequency が 0 以下の場合にバリデーションエラーが発生することを確認する."""
        with pytest.raises(ValidationError):
            GradientTrackingConfig(record_frequency=0)


class TestConfusionMatrixConfig:
    """ConfusionMatrixConfig のテスト."""

    def test_model_validate_applies_partial_values(self) -> None:
        """一部キーだけ指定した場合の補完動作を確認する."""
        config = ConfusionMatrixConfig.model_validate(
            {
                "title": "My Matrix",
                "figsize": (10, 8),
            }
        )

        assert config.title == "My Matrix"
        assert config.figsize == (10, 8)
        assert config.cmap == "Blues"
        assert config.xlabel == "Predicted Label"

    def test_invalid_fontsize_raises_error(self) -> None:
        """fontsize が 0 以下の場合にバリデーションエラーが発生することを確認する."""
        with pytest.raises(ValidationError):
            ConfusionMatrixConfig(fontsize=0)


class TestOptunaConfig:
    """OptunaConfig のテスト."""

    def test_model_validate_defaults_and_custom_values(self) -> None:
        """デフォルト値とカスタム値が正しく反映されることを確認する."""
        config = OptunaConfig.model_validate(
            {
                "search_space": {"learning_rate": {"type": "float"}},
                "n_trials": 30,
                "sampler": "RandomSampler",
                "storage": "sqlite:///study.db",
            }
        )

        assert "learning_rate" in config.search_space
        assert config.n_trials == 30
        assert config.n_jobs == 1
        assert config.sampler == "RandomSampler"
        assert config.storage == "sqlite:///study.db"

    def test_invalid_n_trials_raises_error(self) -> None:
        """n_trials が 0 以下の場合にバリデーションエラーが発生することを確認する."""
        with pytest.raises(ValidationError):
            OptunaConfig(n_trials=0)
