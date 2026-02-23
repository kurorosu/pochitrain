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


class TestTrainingConfiguratorLayerGroup:
    """TrainingConfigurator._get_layer_group() のテスト."""

    def test_get_layer_group(self, configurator):
        """層グループ名の取得テスト."""
        assert configurator._get_layer_group("conv1.weight") == "conv1"
        assert configurator._get_layer_group("bn1.weight") == "bn1"
        assert configurator._get_layer_group("layer1.0.conv1.weight") == "layer1"
        assert configurator._get_layer_group("layer2.1.bn2.bias") == "layer2"
        assert configurator._get_layer_group("layer3.0.downsample.0.weight") == "layer3"
        assert configurator._get_layer_group("layer4.1.conv2.weight") == "layer4"
        assert configurator._get_layer_group("fc.weight") == "fc"
        assert configurator._get_layer_group("unknown.weight") == "other"


class TestTrainingConfiguratorBuildParamGroups:
    """TrainingConfigurator._build_layer_wise_param_groups() のテスト."""

    def test_build_layer_wise_param_groups(self, configurator, model):
        """パラメータグループ構築のテスト."""
        lr_config = {
            "layer_rates": {
                "conv1": 0.0001,
                "layer1": 0.0002,
                "fc": 0.01,
            }
        }

        param_groups = configurator._build_layer_wise_param_groups(
            model, 0.001, lr_config
        )

        assert len(param_groups) > 0
        for group in param_groups:
            assert "params" in group
            assert "lr" in group
            assert "layer_name" in group
            assert isinstance(group["params"], list)
            assert isinstance(group["lr"], float)
            assert group["lr"] > 0
