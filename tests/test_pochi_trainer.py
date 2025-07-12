"""
PochiTrainerの基本テスト
"""

import tempfile

import pytest
import torch

from pochitrain.pochi_trainer import PochiTrainer


def test_pochi_trainer_init():
    """PochiTrainerの初期化テスト"""
    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = PochiTrainer(
            model_name="resnet18",
            num_classes=10,
            pretrained=False,  # テスト用に高速化
            device="cpu",  # テスト用にCPUを指定
            work_dir=temp_dir,  # テスト用に一時ディレクトリを使用
        )

        assert trainer.model is not None
        assert trainer.device == torch.device("cpu")
        assert trainer.epoch == 0
        assert trainer.best_accuracy == 0.0


def test_pochi_trainer_setup_training():
    """PochiTrainerの訓練設定テスト"""
    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = PochiTrainer(
            model_name="resnet18",
            num_classes=10,
            pretrained=False,
            device="cpu",
            work_dir=temp_dir,
        )

        trainer.setup_training(learning_rate=0.001, optimizer_name="Adam")

        assert trainer.optimizer is not None
        assert trainer.criterion is not None


def test_pochi_trainer_invalid_model():
    """無効なモデル名のテスト"""
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(ValueError):
            PochiTrainer(
                model_name="invalid_model",
                num_classes=10,
                pretrained=False,
                device="cpu",
                work_dir=temp_dir,
            )


def test_pochi_trainer_invalid_optimizer():
    """無効な最適化器のテスト"""
    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = PochiTrainer(
            model_name="resnet18",
            num_classes=10,
            pretrained=False,
            device="cpu",
            work_dir=temp_dir,
        )

        with pytest.raises(ValueError):
            trainer.setup_training(
                learning_rate=0.001, optimizer_name="InvalidOptimizer"
            )
