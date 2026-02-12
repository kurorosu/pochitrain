"""sub_configs のテスト."""

from pochitrain.config.sub_configs import (
    ConfusionMatrixConfig,
    EarlyStoppingConfig,
    GradientTrackingConfig,
    LayerWiseLRConfig,
    OptunaConfig,
)


class TestEarlyStoppingConfig:
    """EarlyStoppingConfig のテスト."""

    def test_from_dict_none_uses_defaults(self) -> None:
        """None 入力でデフォルト値が使われることを確認する."""
        config = EarlyStoppingConfig.from_dict(None)

        assert config.enabled is False
        assert config.patience == 10
        assert config.min_delta == 0.0
        assert config.monitor == "val_accuracy"


class TestLayerWiseLRConfig:
    """LayerWiseLRConfig のテスト."""

    def test_from_dict_builds_nested_graph_config(self) -> None:
        """graph_config を含む辞書を正しく変換できることを確認する."""
        config = LayerWiseLRConfig.from_dict(
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

    def test_from_dict_overrides_defaults(self) -> None:
        """カスタム設定でデフォルトが上書きされることを確認する."""
        config = GradientTrackingConfig.from_dict(
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


class TestConfusionMatrixConfig:
    """ConfusionMatrixConfig のテスト."""

    def test_from_dict_applies_partial_values(self) -> None:
        """一部キーだけ指定した場合の補完動作を確認する."""
        config = ConfusionMatrixConfig.from_dict(
            {
                "title": "My Matrix",
                "figsize": (10, 8),
            }
        )

        assert config.title == "My Matrix"
        assert config.figsize == (10, 8)
        assert config.cmap == "Blues"
        assert config.xlabel == "Predicted Label"


class TestOptunaConfig:
    """OptunaConfig のテスト."""

    def test_from_dict_defaults_and_custom_values(self) -> None:
        """デフォルト値とカスタム値が正しく反映されることを確認する."""
        config = OptunaConfig.from_dict(
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
