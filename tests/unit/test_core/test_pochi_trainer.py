"""
PochiTrainerの基本テスト
"""

import tempfile
from unittest.mock import patch

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


@pytest.mark.parametrize(
    "model_name, raises",
    [
        ("resnet18", False),
        ("invalid_model", True),
    ],
)
def test_pochi_trainer_init_model_validation(tmp_path, model_name, raises):
    """モデル名のバリデーションテスト."""
    if raises:
        with pytest.raises(ValueError):
            PochiTrainer(
                model_name=model_name,
                num_classes=2,
                device="cpu",
                work_dir=str(tmp_path),
            )
    else:
        PochiTrainer(
            model_name=model_name, num_classes=2, device="cpu", work_dir=str(tmp_path)
        )


@pytest.mark.parametrize(
    "optimizer_name, raises",
    [
        ("Adam", False),
        ("InvalidOptimizer", True),
    ],
)
def test_pochi_trainer_setup_training_optimizer_validation(
    tmp_path, optimizer_name, raises
):
    """最適化器名のバリデーションテスト."""
    trainer = PochiTrainer(
        model_name="resnet18", num_classes=2, device="cpu", work_dir=str(tmp_path)
    )
    if raises:
        with pytest.raises(ValueError):
            trainer.setup_training(learning_rate=0.001, optimizer_name=optimizer_name)
    else:
        trainer.setup_training(learning_rate=0.001, optimizer_name=optimizer_name)


@pytest.mark.parametrize(
    "weights, num_classes, match",
    [
        ([1.0, 2.0], 3, "クラス重みの長さ.*がクラス数.*と一致しません"),
        (None, 2, None),
        ([1.0, 0.5], 2, None),
    ],
)
def test_pochi_trainer_class_weights_validation(tmp_path, weights, num_classes, match):
    """クラス重みのバリデーションと適用テスト."""
    trainer = PochiTrainer(
        model_name="resnet18",
        num_classes=num_classes,
        pretrained=False,
        device="cpu",
        work_dir=str(tmp_path),
    )

    if match:
        with pytest.raises(ValueError, match=match):
            trainer.setup_training(
                learning_rate=0.001,
                optimizer_name="Adam",
                class_weights=weights,
                num_classes=num_classes,
            )
    else:
        trainer.setup_training(
            learning_rate=0.001,
            optimizer_name="Adam",
            class_weights=weights,
            num_classes=num_classes,
        )
        assert trainer.criterion is not None
        if weights is None:
            assert trainer.criterion.weight is None
        else:
            expected = torch.tensor(weights, dtype=torch.float32)
            assert torch.allclose(trainer.criterion.weight.cpu(), expected)


def test_pochi_trainer_train_without_setup_raises():
    """setup_training() 未実行で train() を呼ぶと RuntimeError になることをテスト"""
    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = PochiTrainer(
            model_name="resnet18",
            num_classes=10,
            pretrained=False,
            device="cpu",
            work_dir=temp_dir,
        )

        with pytest.raises(RuntimeError, match="setup_training"):
            trainer.train(train_loader=[], epochs=1)


def test_pochi_trainer_train_one_epoch_without_setup_raises():
    """setup_training() 未実行で train_one_epoch() が RuntimeError になることをテスト"""
    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = PochiTrainer(
            model_name="resnet18",
            num_classes=10,
            pretrained=False,
            device="cpu",
            work_dir=temp_dir,
        )

        with pytest.raises(RuntimeError, match="setup_training"):
            trainer.train_one_epoch(epoch=1, train_loader=[])


def test_pochi_trainer_train_one_epoch_sets_epoch():
    """train_one_epoch() が epoch を更新して訓練を実行することをテスト"""
    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = PochiTrainer(
            model_name="resnet18",
            num_classes=10,
            pretrained=False,
            device="cpu",
            work_dir=temp_dir,
        )
        trainer.setup_training(learning_rate=0.001, optimizer_name="Adam")

        metrics = trainer.train_one_epoch(epoch=3, train_loader=[])

        assert trainer.epoch == 3
        assert metrics["loss"] == 0.0
        assert metrics["accuracy"] == 0.0


def test_pochi_trainer_train_without_workspace_skips_visualization_dir(monkeypatch):
    """workspace未作成時に可視化ディレクトリを参照しないことをテスト"""
    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = PochiTrainer(
            model_name="resnet18",
            num_classes=10,
            pretrained=False,
            device="cpu",
            work_dir=temp_dir,
            create_workspace=False,
        )
        trainer.setup_training(learning_rate=0.001, optimizer_name="Adam")

        def fail_if_called():
            raise AssertionError("get_visualization_dir should not be called")

        monkeypatch.setattr(
            trainer.workspace_manager, "get_visualization_dir", fail_if_called
        )

        with patch(
            "pochitrain.pochi_trainer.TrainingLoop.run",
            return_value=(0, trainer.best_accuracy),
        ) as mock_run:
            trainer.train(train_loader=[], epochs=1)

        mock_run.assert_called_once()


def test_pochi_trainer_train_passes_initial_best_accuracy():
    """train() が現在のbest_accuracyをTrainingLoopへ渡すことをテスト"""
    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = PochiTrainer(
            model_name="resnet18",
            num_classes=10,
            pretrained=False,
            device="cpu",
            work_dir=temp_dir,
        )
        trainer.setup_training(learning_rate=0.001, optimizer_name="Adam")
        trainer.best_accuracy = 88.8

        with patch(
            "pochitrain.pochi_trainer.TrainingLoop.run",
            return_value=(1, 90.0),
        ) as mock_run:
            trainer.train(train_loader=[], epochs=1)

        assert mock_run.call_args.kwargs["initial_best_accuracy"] == pytest.approx(88.8)
        assert trainer.best_accuracy == pytest.approx(90.0)
