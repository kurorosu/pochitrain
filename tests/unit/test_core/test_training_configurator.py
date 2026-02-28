"""
TrainingConfigurator のテスト.
"""

import pytest
import torch

from pochitrain.models.pochi_models import create_model
from pochitrain.training.training_configurator import (
    TrainingComponents,
    TrainingConfigurator,
)


@pytest.fixture
def configurator():
    """テスト用の TrainingConfigurator インスタンスを作成."""
    import logging

    logger = logging.getLogger("test_training_configurator")
    return TrainingConfigurator(device=torch.device("cpu"), logger=logger)


@pytest.fixture
def model():
    """テスト用の ResNet18 モデルを作成."""
    return create_model("resnet18", num_classes=4, pretrained=False)


class TestTrainingConfiguratorConfigure:
    """TrainingConfigurator.configure() のテスト."""

    def test_configure_adam(self, configurator, model):
        """Adam optimizer の構築テスト."""
        components = configurator.configure(
            model=model,
            learning_rate=0.001,
            optimizer_name="Adam",
        )

        assert isinstance(components, TrainingComponents)
        assert isinstance(components.optimizer, torch.optim.Adam)
        assert components.scheduler is None
        assert isinstance(components.criterion, torch.nn.CrossEntropyLoss)
        assert components.enable_layer_wise_lr is False
        assert components.base_learning_rate == 0.001

    def test_configure_adamw(self, configurator, model):
        """AdamW optimizer の構築テスト."""
        components = configurator.configure(
            model=model,
            learning_rate=0.001,
            optimizer_name="AdamW",
        )

        assert isinstance(components.optimizer, torch.optim.AdamW)

    def test_configure_sgd(self, configurator, model):
        """SGD optimizer の構築テスト."""
        components = configurator.configure(
            model=model,
            learning_rate=0.01,
            optimizer_name="SGD",
        )

        assert isinstance(components.optimizer, torch.optim.SGD)
        assert components.base_learning_rate == 0.01

    def test_configure_with_scheduler(self, configurator, model):
        """scheduler 付き構築テスト."""
        components = configurator.configure(
            model=model,
            learning_rate=0.001,
            optimizer_name="Adam",
            scheduler_name="StepLR",
            scheduler_params={"step_size": 10, "gamma": 0.1},
        )

        assert components.scheduler is not None
        assert isinstance(components.scheduler, torch.optim.lr_scheduler.StepLR)

    def test_configure_with_all_schedulers(self, configurator, model):
        """全スケジューラーの構築テスト."""
        scheduler_configs = [
            ("StepLR", {"step_size": 10, "gamma": 0.1}),
            ("MultiStepLR", {"milestones": [10, 20], "gamma": 0.1}),
            ("CosineAnnealingLR", {"T_max": 50}),
            ("ExponentialLR", {"gamma": 0.95}),
            ("LinearLR", {"start_factor": 0.1, "total_iters": 10}),
        ]

        for scheduler_name, scheduler_params in scheduler_configs:
            components = configurator.configure(
                model=model,
                learning_rate=0.001,
                optimizer_name="Adam",
                scheduler_name=scheduler_name,
                scheduler_params=scheduler_params,
            )
            assert components.scheduler is not None, f"{scheduler_name} の構築に失敗"

    def test_configure_with_class_weights(self, configurator, model):
        """クラス重み設定テスト."""
        class_weights = [1.0, 2.0, 0.5, 1.5]
        components = configurator.configure(
            model=model,
            learning_rate=0.001,
            optimizer_name="Adam",
            class_weights=class_weights,
            num_classes=4,
        )

        assert components.criterion.weight is not None
        expected = torch.tensor(class_weights, dtype=torch.float32)
        assert torch.allclose(components.criterion.weight, expected)

    def test_configure_without_class_weights(self, configurator, model):
        """クラス重みなし設定テスト."""
        components = configurator.configure(
            model=model,
            learning_rate=0.001,
            optimizer_name="Adam",
        )

        assert components.criterion.weight is None

    def test_configure_class_weights_mismatch(self, configurator, model):
        """クラス重みとクラス数の不整合テスト."""
        with pytest.raises(
            ValueError, match="クラス重みの長さ.*がクラス数.*と一致しません"
        ):
            configurator.configure(
                model=model,
                learning_rate=0.001,
                optimizer_name="Adam",
                class_weights=[1.0, 2.0],
                num_classes=4,
            )

    def test_configure_layer_wise_lr(self, configurator, model):
        """層別学習率構築テスト."""
        layer_wise_lr_config = {
            "layer_rates": {
                "conv1": 0.0001,
                "layer1": 0.0002,
                "fc": 0.01,
            },
            "graph_config": {"show_legend": True},
        }

        components = configurator.configure(
            model=model,
            learning_rate=0.001,
            optimizer_name="SGD",
            enable_layer_wise_lr=True,
            layer_wise_lr_config=layer_wise_lr_config,
        )

        assert components.enable_layer_wise_lr is True
        assert components.base_learning_rate == 0.001
        assert components.layer_wise_lr_config == layer_wise_lr_config
        assert components.layer_wise_lr_graph_config == {"show_legend": True}
        assert len(components.optimizer.param_groups) > 1

    def test_invalid_optimizer_raises(self, configurator, model):
        """不正 optimizer 名でエラー."""
        with pytest.raises(ValueError, match="サポートされていない最適化器"):
            configurator.configure(
                model=model,
                learning_rate=0.001,
                optimizer_name="InvalidOptimizer",
            )

    def test_invalid_scheduler_raises(self, configurator, model):
        """不正 scheduler 名でエラー."""
        with pytest.raises(ValueError, match="サポートされていないスケジューラー"):
            configurator.configure(
                model=model,
                learning_rate=0.001,
                optimizer_name="Adam",
                scheduler_name="InvalidScheduler",
                scheduler_params={"step_size": 10},
            )

    def test_scheduler_without_params_raises(self, configurator, model):
        """スケジューラーパラメータなしでエラー."""
        with pytest.raises(ValueError, match="scheduler_paramsが必須"):
            configurator.configure(
                model=model,
                learning_rate=0.001,
                optimizer_name="Adam",
                scheduler_name="StepLR",
                scheduler_params=None,
            )


class TestTrainingConfiguratorLayerWiseLrDetails:
    """層別学習率の configure() 経由での詳細検証テスト."""

    def test_configure_layer_wise_lr_assigns_specified_rates(self, configurator, model):
        """configure() で指定した層別学習率が optimizer の各グループに反映されることを検証."""
        layer_rates = {
            "conv1": 0.0001,
            "layer1": 0.0002,
            "fc": 0.01,
        }
        components = configurator.configure(
            model=model,
            learning_rate=0.001,
            optimizer_name="SGD",
            enable_layer_wise_lr=True,
            layer_wise_lr_config={"layer_rates": layer_rates},
        )

        group_lrs = {
            g["layer_name"]: g["lr"] for g in components.optimizer.param_groups
        }
        for layer_name, expected_lr in layer_rates.items():
            assert group_lrs[layer_name] == pytest.approx(expected_lr)

    def test_configure_layer_wise_lr_uses_base_lr_for_unspecified_layers(
        self, configurator, model
    ):
        """configure() で未指定の層にはベース学習率が適用されることを検証."""
        components = configurator.configure(
            model=model,
            learning_rate=0.005,
            optimizer_name="SGD",
            enable_layer_wise_lr=True,
            layer_wise_lr_config={"layer_rates": {"fc": 0.01}},
        )

        for group in components.optimizer.param_groups:
            if group["layer_name"] != "fc":
                assert group["lr"] == pytest.approx(0.005)
