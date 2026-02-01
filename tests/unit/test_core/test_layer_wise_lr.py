"""
層別学習率機能のテスト.

層別学習率の設定、パラメータグループ作成、メトリクス記録の動作をテストします。
"""

from unittest.mock import patch

import pytest

from pochitrain import PochiTrainer


class TestLayerWiseLR:
    """層別学習率機能のテストクラス."""

    @pytest.fixture
    def trainer(self):
        """テスト用のPochiTrainerインスタンスを作成."""
        return PochiTrainer(
            model_name="resnet18",
            num_classes=4,
            device="cpu",
            pretrained=False,
            create_workspace=False,
        )

    def test_layer_wise_lr_disabled(self, trainer):
        """層別学習率が無効の場合のテスト."""
        trainer.setup_training(
            learning_rate=0.001,
            optimizer_name="SGD",
            enable_layer_wise_lr=False,
        )

        # 通常のパラメータグループが作成されることを確認
        assert len(trainer.optimizer.param_groups) == 1
        assert trainer.optimizer.param_groups[0]["lr"] == 0.001
        assert not trainer.enable_layer_wise_lr

    def test_layer_wise_lr_enabled(self, trainer):
        """層別学習率が有効の場合のテスト."""
        layer_wise_lr_config = {
            "layer_rates": {
                "conv1": 0.0001,
                "bn1": 0.0001,
                "layer1": 0.0002,
                "layer2": 0.0005,
                "layer3": 0.001,
                "layer4": 0.002,
                "fc": 0.01,
            }
        }

        trainer.setup_training(
            learning_rate=0.001,
            optimizer_name="SGD",
            enable_layer_wise_lr=True,
            layer_wise_lr_config=layer_wise_lr_config,
        )

        # 複数のパラメータグループが作成されることを確認
        assert len(trainer.optimizer.param_groups) > 1
        assert trainer.enable_layer_wise_lr
        assert trainer.base_learning_rate == 0.001

    def test_get_layer_group(self, trainer):
        """層グループ名の取得テスト (TrainingConfigurator 経由)."""
        configurator = trainer.training_configurator
        assert configurator._get_layer_group("conv1.weight") == "conv1"
        assert configurator._get_layer_group("bn1.weight") == "bn1"
        assert configurator._get_layer_group("layer1.0.conv1.weight") == "layer1"
        assert configurator._get_layer_group("layer2.1.bn2.bias") == "layer2"
        assert configurator._get_layer_group("layer3.0.downsample.0.weight") == "layer3"
        assert configurator._get_layer_group("layer4.1.conv2.weight") == "layer4"
        assert configurator._get_layer_group("fc.weight") == "fc"
        assert configurator._get_layer_group("unknown.weight") == "other"

    def test_build_layer_wise_param_groups(self, trainer):
        """パラメータグループ構築のテスト (TrainingConfigurator 経由)."""
        layer_wise_lr_config = {
            "layer_rates": {
                "conv1": 0.0001,
                "layer1": 0.0002,
                "fc": 0.01,
            }
        }

        param_groups = trainer.training_configurator._build_layer_wise_param_groups(
            trainer.model, 0.001, layer_wise_lr_config
        )

        # パラメータグループが作成されることを確認
        assert len(param_groups) > 0

        # 各グループに必要な情報が含まれることを確認
        for group in param_groups:
            assert "params" in group
            assert "lr" in group
            assert "layer_name" in group
            assert isinstance(group["params"], list)
            assert isinstance(group["lr"], float)
            assert group["lr"] > 0

    def test_get_base_learning_rate(self, trainer):
        """基本学習率取得のテスト."""
        # 層別学習率無効時
        trainer.setup_training(
            learning_rate=0.001,
            optimizer_name="SGD",
            enable_layer_wise_lr=False,
        )
        lr = trainer._get_base_learning_rate()
        assert lr == 0.001
        assert not trainer.is_layer_wise_lr_enabled()

        # 層別学習率有効時
        layer_wise_lr_config = {
            "layer_rates": {
                "fc": 0.01,
            }
        }
        trainer.setup_training(
            learning_rate=0.002,
            optimizer_name="SGD",
            enable_layer_wise_lr=True,
            layer_wise_lr_config=layer_wise_lr_config,
        )
        lr = trainer._get_base_learning_rate()
        assert lr == 0.002  # 設定ファイルの固定値
        assert trainer.is_layer_wise_lr_enabled()

    def test_layer_wise_lr_with_scheduler(self, trainer):
        """層別学習率とスケジューラーの組み合わせテスト."""
        layer_wise_lr_config = {
            "layer_rates": {
                "conv1": 0.0001,
                "fc": 0.01,
            }
        }

        trainer.setup_training(
            learning_rate=0.001,
            optimizer_name="SGD",
            scheduler_name="StepLR",
            scheduler_params={"step_size": 10, "gamma": 0.1},
            enable_layer_wise_lr=True,
            layer_wise_lr_config=layer_wise_lr_config,
        )

        # スケジューラーが設定されることを確認
        assert trainer.scheduler is not None
        assert trainer.enable_layer_wise_lr

    @patch(
        "pochitrain.training.training_configurator.TrainingConfigurator._log_layer_wise_lr"
    )
    def test_log_layer_wise_lr_called(self, mock_log, trainer):
        """層別学習率ログ出力が呼ばれることのテスト."""
        layer_wise_lr_config = {
            "layer_rates": {
                "fc": 0.01,
            }
        }

        trainer.setup_training(
            learning_rate=0.001,
            optimizer_name="SGD",
            enable_layer_wise_lr=True,
            layer_wise_lr_config=layer_wise_lr_config,
        )

        # ログ出力メソッドが呼ばれることを確認
        mock_log.assert_called_once()

    def test_layer_wise_lr_validation_error(self, trainer):
        """不正な層別学習率設定でのエラーテスト."""
        # layer_ratesが空の場合
        layer_wise_lr_config = {"layer_rates": {}}

        # エラーが発生しないことを確認（バリデーションは別途テスト）
        trainer.setup_training(
            learning_rate=0.001,
            optimizer_name="SGD",
            enable_layer_wise_lr=True,
            layer_wise_lr_config=layer_wise_lr_config,
        )

        assert trainer.enable_layer_wise_lr
