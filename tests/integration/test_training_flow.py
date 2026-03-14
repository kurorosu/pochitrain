"""訓練フローのエンドツーエンド統合テスト.

PochiTrainerを使った1エポック訓練の全体フローを検証する.
"""

import torch
import torchvision.transforms as transforms

from pochitrain import PochiTrainer
from pochitrain.pochi_dataset import create_data_loaders


class TestTrainingFlow:
    """エンドツーエンド訓練フローの統合テスト."""

    def test_single_epoch_training(self, tmp_path, create_dummy_train_val):
        """1エポック訓練が正常に完了し, モデルが更新されること."""
        train_root, val_root = create_dummy_train_val()

        trainer = PochiTrainer(
            model_name="resnet18",
            num_classes=2,
            device="cpu",
            pretrained=False,
            work_dir=str(tmp_path / "work_dirs"),
        )

        trainer.setup_training(
            learning_rate=0.01,
            optimizer_name="SGD",
        )

        train_transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ]
        )
        val_transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ]
        )

        train_loader, val_loader, classes = create_data_loaders(
            train_root=train_root,
            val_root=val_root,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
            train_transform=train_transform,
            val_transform=val_transform,
        )

        params_before = {
            name: param.clone()
            for name, param in trainer.model.named_parameters()
            if param.requires_grad
        }

        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=1,
        )

        # 訓練後にパラメータが更新されていること
        params_changed = False
        for name, param in trainer.model.named_parameters():
            if param.requires_grad and not torch.equal(param, params_before[name]):
                params_changed = True
                break
        assert params_changed, "訓練後にモデルパラメータが更新されていない"

        assert trainer.epoch == 1

    def test_training_with_scheduler(self, tmp_path, create_dummy_train_val):
        """スケジューラー付き訓練が正常に完了すること."""
        train_root, val_root = create_dummy_train_val()

        trainer = PochiTrainer(
            model_name="resnet18",
            num_classes=2,
            device="cpu",
            pretrained=False,
            work_dir=str(tmp_path / "work_dirs"),
        )

        trainer.setup_training(
            learning_rate=0.01,
            optimizer_name="Adam",
            scheduler_name="StepLR",
            scheduler_params={"step_size": 1, "gamma": 0.5},
        )

        initial_lr = trainer.optimizer.param_groups[0]["lr"]

        train_transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ]
        )
        val_transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ]
        )

        train_loader, val_loader, classes = create_data_loaders(
            train_root=train_root,
            val_root=val_root,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
            train_transform=train_transform,
            val_transform=val_transform,
        )

        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=2,
        )

        # スケジューラーにより学習率が低下していること
        final_lr = trainer.optimizer.param_groups[0]["lr"]
        assert final_lr < initial_lr

        assert trainer.epoch == 2

    def test_training_saves_checkpoint(self, tmp_path, create_dummy_train_val):
        """訓練後にチェックポイントが保存されること."""
        train_root, val_root = create_dummy_train_val()

        work_dir = tmp_path / "work_dirs"
        trainer = PochiTrainer(
            model_name="resnet18",
            num_classes=2,
            device="cpu",
            pretrained=False,
            work_dir=str(work_dir),
        )

        trainer.setup_training(
            learning_rate=0.01,
            optimizer_name="SGD",
        )

        train_transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ]
        )
        val_transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ]
        )

        train_loader, val_loader, classes = create_data_loaders(
            train_root=train_root,
            val_root=val_root,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
            train_transform=train_transform,
            val_transform=val_transform,
        )

        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=1,
        )

        # モデルディレクトリにチェックポイントが保存されていること
        model_files = list(trainer.work_dir.glob("*.pth"))
        assert len(model_files) > 0, "チェックポイントが保存されていない"
