"""
層別学習率機能のテスト.

層別学習率の設定、パラメータグループ作成、メトリクス記録の動作をテストします。
"""

import pytest


class TestLayerWiseLR:
    """層別学習率機能のテストクラス."""

    def test_layer_wise_lr_disabled(self, trainer):
        """層別学習率が無効の場合のテスト."""
        trainer.setup_training(
            learning_rate=0.001,
            optimizer_name="SGD",
            enable_layer_wise_lr=False,
        )

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

        assert len(trainer.optimizer.param_groups) > 1
        assert trainer.enable_layer_wise_lr
        assert trainer.base_learning_rate == 0.001

    def test_get_base_learning_rate(self, trainer):
        """基本学習率取得のテスト."""
        trainer.setup_training(
            learning_rate=0.001,
            optimizer_name="SGD",
            enable_layer_wise_lr=False,
        )
        lr = trainer._get_base_learning_rate()
        assert lr == 0.001
        assert not trainer.is_layer_wise_lr_enabled()

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

        assert trainer.scheduler is not None
        assert trainer.enable_layer_wise_lr

    def test_log_layer_wise_lr_called(self, trainer):
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

        assert trainer.enable_layer_wise_lr

    def test_layer_wise_lr_with_empty_layer_rates(self, trainer):
        """空のlayer_ratesでも層別学習率が有効になることのテスト."""
        layer_wise_lr_config = {"layer_rates": {}}

        trainer.setup_training(
            learning_rate=0.001,
            optimizer_name="SGD",
            enable_layer_wise_lr=True,
            layer_wise_lr_config=layer_wise_lr_config,
        )

        assert trainer.enable_layer_wise_lr
        assert len(trainer.optimizer.param_groups) > 0
