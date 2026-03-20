"""layer_wise_lr パッケージのテスト.

ResNetLayerGrouper, ParamGroupBuilder を実際の PyTorch モデルで検証する古典派テスト.
"""

import logging

import pytest
import torch
import torch.nn as nn

from pochitrain.models.pochi_models import create_model
from pochitrain.training.layer_wise_lr import (
    ILayerGrouper,
    ParamGroupBuilder,
    ResNetLayerGrouper,
)

# --- ResNetLayerGrouper ---


class TestResNetLayerGrouper:
    """ResNetLayerGrouper のテスト."""

    @pytest.fixture
    def grouper(self) -> ResNetLayerGrouper:
        """テスト用の ResNetLayerGrouper インスタンス."""
        return ResNetLayerGrouper()

    def test_layer1_params(self, grouper: ResNetLayerGrouper):
        """layer1 を含むパラメータ名は 'layer1' を返す."""
        assert grouper.get_group("layer1.0.conv1.weight") == "layer1"

    def test_layer2_params(self, grouper: ResNetLayerGrouper):
        """layer2 を含むパラメータ名は 'layer2' を返す."""
        assert grouper.get_group("layer2.1.bn2.weight") == "layer2"

    def test_layer3_params(self, grouper: ResNetLayerGrouper):
        """layer3 を含むパラメータ名は 'layer3' を返す."""
        assert grouper.get_group("layer3.0.conv2.weight") == "layer3"

    def test_layer4_params(self, grouper: ResNetLayerGrouper):
        """layer4 を含むパラメータ名は 'layer4' を返す."""
        assert grouper.get_group("layer4.0.conv1.weight") == "layer4"

    def test_conv1_params(self, grouper: ResNetLayerGrouper):
        """conv1 を含むパラメータ名は 'conv1' を返す."""
        assert grouper.get_group("conv1.weight") == "conv1"

    def test_bn1_params(self, grouper: ResNetLayerGrouper):
        """bn1 を含むパラメータ名は 'bn1' を返す."""
        assert grouper.get_group("bn1.weight") == "bn1"

    def test_fc_params(self, grouper: ResNetLayerGrouper):
        """fc を含むパラメータ名は 'fc' を返す."""
        assert grouper.get_group("fc.weight") == "fc"

    def test_unknown_param_returns_other(self, grouper: ResNetLayerGrouper):
        """該当しないパラメータ名は 'other' を返す."""
        assert grouper.get_group("some_unknown_module.weight") == "other"

    def test_implements_interface(self):
        """ResNetLayerGrouper は ILayerGrouper を実装している."""
        assert issubclass(ResNetLayerGrouper, ILayerGrouper)

    def test_all_resnet18_params_are_grouped(self, grouper: ResNetLayerGrouper):
        """ResNet18 の全パラメータが 'other' 以外にグルーピングされる."""
        model = create_model("resnet18", num_classes=4, pretrained=False)
        for name, _ in model.named_parameters():
            group = grouper.get_group(name)
            assert group != "other", f"パラメータ '{name}' が 'other' に分類された"


# --- ParamGroupBuilder ---


class TestParamGroupBuilder:
    """ParamGroupBuilder のテスト."""

    @pytest.fixture
    def builder(self, logger: logging.Logger) -> ParamGroupBuilder:
        """テスト用の ParamGroupBuilder インスタンス."""
        return ParamGroupBuilder(ResNetLayerGrouper(), logger)

    @pytest.fixture
    def model(self) -> nn.Module:
        """テスト用の ResNet18 モデル."""
        return create_model("resnet18", num_classes=4, pretrained=False)

    def test_build_creates_multiple_groups(
        self, builder: ParamGroupBuilder, model: nn.Module
    ):
        """build() は複数のパラメータグループを作成する."""
        groups = builder.build(model, base_lr=0.001, layer_wise_lr_config={})

        assert len(groups) > 1

    def test_build_assigns_specified_rates(
        self, builder: ParamGroupBuilder, model: nn.Module
    ):
        """指定した層には指定学習率が適用される."""
        config = {"layer_rates": {"conv1": 0.0001, "fc": 0.01}}
        groups = builder.build(model, base_lr=0.001, layer_wise_lr_config=config)

        group_lrs = {g["layer_name"]: g["lr"] for g in groups}
        assert group_lrs["conv1"] == pytest.approx(0.0001)
        assert group_lrs["fc"] == pytest.approx(0.01)

    def test_build_uses_base_lr_for_unspecified(
        self, builder: ParamGroupBuilder, model: nn.Module
    ):
        """未指定の層にはベース学習率が適用される."""
        config = {"layer_rates": {"fc": 0.01}}
        groups = builder.build(model, base_lr=0.005, layer_wise_lr_config=config)

        for group in groups:
            if group["layer_name"] != "fc":
                assert group["lr"] == pytest.approx(0.005)

    def test_build_skips_frozen_params(
        self, builder: ParamGroupBuilder, model: nn.Module
    ):
        """requires_grad=False のパラメータはグループに含まれない."""
        # conv1 のパラメータを凍結
        for name, param in model.named_parameters():
            if "conv1" in name:
                param.requires_grad = False

        groups = builder.build(model, base_lr=0.001, layer_wise_lr_config={})

        group_names = {g["layer_name"] for g in groups}
        assert "conv1" not in group_names

    def test_build_all_params_included(
        self, builder: ParamGroupBuilder, model: nn.Module
    ):
        """全 trainable パラメータがいずれかのグループに含まれる."""
        groups = builder.build(model, base_lr=0.001, layer_wise_lr_config={})

        total_in_groups = sum(sum(p.numel() for p in g["params"]) for g in groups)
        total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total_in_groups == total_trainable

    def test_build_with_empty_config(
        self, builder: ParamGroupBuilder, model: nn.Module
    ):
        """空の config でも正常に動作する."""
        groups = builder.build(model, base_lr=0.001, layer_wise_lr_config={})

        for group in groups:
            assert group["lr"] == pytest.approx(0.001)

    def test_each_group_has_layer_name(
        self, builder: ParamGroupBuilder, model: nn.Module
    ):
        """各グループに layer_name キーが含まれる."""
        groups = builder.build(model, base_lr=0.001, layer_wise_lr_config={})

        for group in groups:
            assert "layer_name" in group
            assert isinstance(group["layer_name"], str)

    def test_log_param_groups(self, builder: ParamGroupBuilder, model: nn.Module):
        """log_param_groups() がエラーなく実行される."""
        groups = builder.build(model, base_lr=0.001, layer_wise_lr_config={})

        # エラーなく実行できることを確認
        builder.log_param_groups(groups)
