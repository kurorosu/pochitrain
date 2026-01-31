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


def test_pochi_trainer_cudnn_benchmark_disabled_on_cpu():
    """CPU環境ではcudnn_benchmarkが無効になることをテスト"""
    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = PochiTrainer(
            model_name="resnet18",
            num_classes=10,
            pretrained=False,
            device="cpu",
            work_dir=temp_dir,
            cudnn_benchmark=True,  # TrueでもCPUでは無効
        )

        assert trainer.cudnn_benchmark is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_pochi_trainer_cudnn_benchmark_enabled_on_cuda():
    """CUDA環境でcudnn_benchmark=Trueが有効になることをテスト"""
    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = PochiTrainer(
            model_name="resnet18",
            num_classes=10,
            pretrained=False,
            device="cuda",
            work_dir=temp_dir,
            cudnn_benchmark=True,
        )

        assert trainer.cudnn_benchmark is True
        assert torch.backends.cudnn.benchmark is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_pochi_trainer_cudnn_benchmark_disabled_on_cuda():
    """CUDA環境でcudnn_benchmark=Falseが無効のままであることをテスト"""
    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = PochiTrainer(
            model_name="resnet18",
            num_classes=10,
            pretrained=False,
            device="cuda",
            work_dir=temp_dir,
            cudnn_benchmark=False,
        )

        assert trainer.cudnn_benchmark is False


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


def test_pochi_trainer_class_weights_cpu():
    """CPU環境でのクラス重み設定テスト"""
    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = PochiTrainer(
            model_name="resnet18",
            num_classes=3,
            pretrained=False,
            device="cpu",
            work_dir=temp_dir,
        )

        # クラス重みを設定
        class_weights = [1.0, 2.0, 0.5]
        trainer.setup_training(
            learning_rate=0.001,
            optimizer_name="Adam",
            class_weights=class_weights,
            num_classes=3,
        )

        assert trainer.criterion is not None
        # 損失関数の重みがCPUデバイスに配置されていることを確認
        assert trainer.criterion.weight.device == torch.device("cpu")
        # 重みの値が正しく設定されていることを確認
        expected_weights = torch.tensor(class_weights, dtype=torch.float32)
        assert torch.allclose(trainer.criterion.weight, expected_weights)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_pochi_trainer_class_weights_cuda():
    """CUDA環境でのクラス重み設定テスト（CUDA利用可能時のみ）"""
    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = PochiTrainer(
            model_name="resnet18",
            num_classes=4,
            pretrained=False,
            device="cuda",
            work_dir=temp_dir,
        )

        # クラス重みを設定
        class_weights = [1.0, 3.0, 1.5, 0.8]
        trainer.setup_training(
            learning_rate=0.001,
            optimizer_name="Adam",
            class_weights=class_weights,
            num_classes=4,
        )

        assert trainer.criterion is not None
        # 損失関数の重みがCUDAデバイスに配置されていることを確認
        assert trainer.criterion.weight.device.type == "cuda"
        # 重みの値が正しく設定されていることを確認
        expected_weights = torch.tensor(class_weights, dtype=torch.float32)
        assert torch.allclose(trainer.criterion.weight.cpu(), expected_weights)


def test_pochi_trainer_class_weights_mismatch():
    """クラス重みとクラス数の不整合テスト"""
    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = PochiTrainer(
            model_name="resnet18",
            num_classes=3,
            pretrained=False,
            device="cpu",
            work_dir=temp_dir,
        )

        # クラス重みの長さがクラス数と一致しない場合
        class_weights = [1.0, 2.0]  # 2つの重みだが、クラス数は3
        with pytest.raises(
            ValueError, match="クラス重みの長さ.*がクラス数.*と一致しません"
        ):
            trainer.setup_training(
                learning_rate=0.001,
                optimizer_name="Adam",
                class_weights=class_weights,
                num_classes=3,
            )


def test_pochi_trainer_class_weights_none():
    """クラス重みなしの場合のテスト"""
    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = PochiTrainer(
            model_name="resnet18",
            num_classes=5,
            pretrained=False,
            device="cpu",
            work_dir=temp_dir,
        )

        trainer.setup_training(
            learning_rate=0.001,
            optimizer_name="Adam",
            class_weights=None,
        )

        assert trainer.criterion is not None
        # 重みが設定されていないことを確認
        assert trainer.criterion.weight is None
