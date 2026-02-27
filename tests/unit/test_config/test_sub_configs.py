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
        assert config.monitor == "val_accuracy"

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"patience": 0}, "patience"),
            ({"min_delta": -0.1}, "min_delta"),
            ({"monitor": "invalid"}, "monitor"),
        ],
    )
    def test_invalid_configs_raise_error(self, kwargs, match) -> None:
        """不正な設定で ValidationError になることを確認する."""
        with pytest.raises(ValidationError, match=match):
            EarlyStoppingConfig(**kwargs)


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
        assert config.graph_config.use_log_scale is False


class TestGradientTrackingConfig:
    """GradientTrackingConfig のテスト."""

    def test_model_validate_overrides_defaults(self) -> None:
        """カスタム設定でデフォルトが上書きされることを確認する."""
        config = GradientTrackingConfig.model_validate(
            {"record_frequency": 3, "aggregation_method": "mean"}
        )
        assert config.record_frequency == 3
        assert config.aggregation_method == "mean"

    def test_invalid_record_frequency_raises_error(self) -> None:
        """record_frequency が 0 以下の場合にバリデーションエラーが発生することを確認する."""
        with pytest.raises(ValidationError, match="record_frequency"):
            GradientTrackingConfig(record_frequency=0)


class TestConfusionMatrixConfig:
    """ConfusionMatrixConfig のテスト."""

    def test_model_validate_applies_partial_values(self) -> None:
        """一部キーだけ指定した場合の補完動作を確認する."""
        config = ConfusionMatrixConfig.model_validate(
            {"title": "My Matrix", "figsize": (10, 8)}
        )
        assert config.title == "My Matrix"
        assert config.figsize == (10, 8)
        assert config.cmap == "Blues"

    def test_invalid_fontsize_raises_error(self) -> None:
        """fontsize が 0 以下の場合にバリデーションエラーが発生することを確認する."""
        with pytest.raises(ValidationError, match="fontsize"):
            ConfusionMatrixConfig(fontsize=0)


class TestOptunaConfig:
    """OptunaConfig のテスト."""

    def test_model_validate_defaults_and_custom_values(self) -> None:
        """デフォルト値とカスタム値が正しく反映されることを確認する."""
        config = OptunaConfig.model_validate(
            {
                "search_space": {"learning_rate": {"type": "float"}},
                "n_trials": 30,
            }
        )
        assert "learning_rate" in config.search_space
        assert config.n_trials == 30

    def test_invalid_n_trials_raises_error(self) -> None:
        """n_trials が 0 以下の場合にバリデーションエラーが発生することを確認する."""
        with pytest.raises(ValidationError, match="n_trials"):
            OptunaConfig(n_trials=0)
