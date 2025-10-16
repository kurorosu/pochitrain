"""ExponentialLR と LinearLR スケジューラーのテスト."""

import pytest
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from pochitrain import PochiTrainer


class TestExponentialLRScheduler:
    """ExponentialLR スケジューラーのテスト."""

    @pytest.fixture
    def trainer(self):
        """トレーナーのフィクスチャ."""
        trainer = PochiTrainer(
            model_name="resnet18",
            num_classes=4,
            device="cpu",
            create_workspace=False,
        )
        return trainer

    @pytest.fixture
    def dummy_loader(self):
        """ダミーデータローダー."""
        X = torch.randn(32, 3, 32, 32)
        y = torch.randint(0, 4, (32,))
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=8)

    def test_exponential_lr_setup(self, trainer):
        """ExponentialLR の設定テスト."""
        trainer.setup_training(
            learning_rate=0.001,
            optimizer_name="SGD",
            scheduler_name="ExponentialLR",
            scheduler_params={"gamma": 0.95},
        )

        assert trainer.scheduler is not None
        assert isinstance(trainer.scheduler, optim.lr_scheduler.ExponentialLR)

    def test_exponential_lr_decay(self, trainer, dummy_loader):
        """ExponentialLR の減衰動作テスト."""
        trainer.setup_training(
            learning_rate=0.001,
            optimizer_name="SGD",
            scheduler_name="ExponentialLR",
            scheduler_params={"gamma": 0.95},
        )

        # 初期学習率を記録
        initial_lr = trainer.optimizer.param_groups[0]["lr"]
        assert abs(initial_lr - 0.001) < 1e-6

        # 1エポック分進める
        trainer.train_epoch(dummy_loader)
        trainer.scheduler.step()

        # 学習率が gamma 倍に減衰したことを確認
        new_lr = trainer.optimizer.param_groups[0]["lr"]
        expected_lr = 0.001 * 0.95
        assert abs(new_lr - expected_lr) < 1e-8

    def test_exponential_lr_with_layer_wise_lr(self, trainer):
        """ExponentialLR と層別学習率の組み合わせテスト."""
        layer_wise_lr_config = {
            "layer_rates": {
                "conv1": 0.0001,
                "fc": 0.01,
            }
        }

        trainer.setup_training(
            learning_rate=0.001,
            optimizer_name="SGD",
            scheduler_name="ExponentialLR",
            scheduler_params={"gamma": 0.95},
            enable_layer_wise_lr=True,
            layer_wise_lr_config=layer_wise_lr_config,
        )

        assert trainer.scheduler is not None
        # パラメータグループが複数存在することを確認
        assert len(trainer.optimizer.param_groups) > 1


class TestLinearLRScheduler:
    """LinearLR スケジューラーのテスト."""

    @pytest.fixture
    def trainer(self):
        """トレーナーのフィクスチャ."""
        trainer = PochiTrainer(
            model_name="resnet18",
            num_classes=4,
            device="cpu",
            create_workspace=False,
        )
        return trainer

    @pytest.fixture
    def dummy_loader(self):
        """ダミーデータローダー."""
        X = torch.randn(32, 3, 32, 32)
        y = torch.randint(0, 4, (32,))
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=8)

    def test_linear_lr_setup(self, trainer):
        """LinearLR の設定テスト."""
        trainer.setup_training(
            learning_rate=0.001,
            optimizer_name="SGD",
            scheduler_name="LinearLR",
            scheduler_params={
                "start_factor": 1.0,
                "end_factor": 0.1,
                "total_iters": 50,
            },
        )

        assert trainer.scheduler is not None
        assert isinstance(trainer.scheduler, optim.lr_scheduler.LinearLR)

    def test_linear_lr_decay(self, trainer, dummy_loader):
        """LinearLR の減衰動作テスト."""
        total_iters = 50
        trainer.setup_training(
            learning_rate=0.001,
            optimizer_name="SGD",
            scheduler_name="LinearLR",
            scheduler_params={
                "start_factor": 1.0,
                "end_factor": 0.1,
                "total_iters": total_iters,
            },
        )

        # 初期学習率を記録
        initial_lr = trainer.optimizer.param_groups[0]["lr"]
        assert abs(initial_lr - 0.001) < 1e-6

        # 複数ステップ進めて学習率が線形に減衰することを確認
        lrs = [initial_lr]
        for _ in range(10):
            trainer.train_epoch(dummy_loader)
            trainer.scheduler.step()
            lrs.append(trainer.optimizer.param_groups[0]["lr"])

        # 学習率が単調減少していることを確認
        for i in range(len(lrs) - 1):
            assert lrs[i] >= lrs[i + 1]

    def test_linear_lr_with_layer_wise_lr(self, trainer):
        """LinearLR と層別学習率の組み合わせテスト."""
        layer_wise_lr_config = {
            "layer_rates": {
                "conv1": 0.0001,
                "fc": 0.01,
            }
        }

        trainer.setup_training(
            learning_rate=0.001,
            optimizer_name="SGD",
            scheduler_name="LinearLR",
            scheduler_params={
                "start_factor": 1.0,
                "end_factor": 0.1,
                "total_iters": 50,
            },
            enable_layer_wise_lr=True,
            layer_wise_lr_config=layer_wise_lr_config,
        )

        assert trainer.scheduler is not None
        # パラメータグループが複数存在することを確認
        assert len(trainer.optimizer.param_groups) > 1


class TestSchedulerInteroperability:
    """各スケジューラーとの相互運用性テスト."""

    @pytest.fixture
    def trainer(self):
        """トレーナーのフィクスチャ."""
        trainer = PochiTrainer(
            model_name="resnet18",
            num_classes=4,
            device="cpu",
            create_workspace=False,
        )
        return trainer

    def test_all_supported_schedulers(self, trainer):
        """すべてのサポート済みスケジューラーが設定できることを確認."""
        schedulers = [
            ("StepLR", {"step_size": 30, "gamma": 0.1}),
            ("MultiStepLR", {"milestones": [30, 60], "gamma": 0.1}),
            ("CosineAnnealingLR", {"T_max": 50}),
            ("ExponentialLR", {"gamma": 0.95}),
            ("LinearLR", {"start_factor": 1.0, "end_factor": 0.1, "total_iters": 50}),
        ]

        for scheduler_name, params in schedulers:
            trainer_tmp = PochiTrainer(
                model_name="resnet18",
                num_classes=4,
                device="cpu",
                create_workspace=False,
            )

            trainer_tmp.setup_training(
                learning_rate=0.001,
                optimizer_name="SGD",
                scheduler_name=scheduler_name,
                scheduler_params=params,
            )

            assert trainer_tmp.scheduler is not None

    def test_unsupported_scheduler_raises_error(self, trainer):
        """未サポートのスケジューラーでエラーが発生することを確認."""
        with pytest.raises(ValueError, match="サポートされていないスケジューラー"):
            trainer.setup_training(
                learning_rate=0.001,
                optimizer_name="SGD",
                scheduler_name="UnsupportedScheduler",
                scheduler_params={"param": 1},
            )
