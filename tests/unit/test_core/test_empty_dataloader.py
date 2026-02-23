import logging
import tempfile

import torch
from torch.utils.data import DataLoader, TensorDataset

from pochitrain.pochi_trainer import PochiTrainer
from pochitrain.training.epoch_runner import EpochRunner
from pochitrain.training.evaluator import Evaluator


def _empty_loader(batch_size: int = 4) -> DataLoader:
    data = torch.empty((0, 3, 224, 224))
    targets = torch.empty((0,), dtype=torch.long)
    dataset = TensorDataset(data, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def test_epoch_runner_with_empty_loader():
    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = PochiTrainer(
            model_name="resnet18",
            num_classes=2,
            pretrained=False,
            device="cpu",
            work_dir=temp_dir,
        )
        trainer.setup_training(
            learning_rate=0.001, optimizer_name="Adam", num_classes=2
        )

        assert trainer.optimizer is not None
        assert trainer.criterion is not None
        epoch_runner = EpochRunner(device=trainer.device, logger=trainer.logger)
        metrics = epoch_runner.run(
            model=trainer.model,
            optimizer=trainer.optimizer,
            criterion=trainer.criterion,
            train_loader=_empty_loader(),
            epoch=0,
        )

        assert metrics["loss"] == 0.0
        assert metrics["accuracy"] == 0.0


def test_epoch_runner_sample_weighted_loss():
    """不均一バッチサイズでサンプル重み付け平均が正しく計算される.

    criterion をモックして固定 loss を返すことで,
    サンプル重み付け平均の計算ロジックを直接検証する.
    バッチ1 (4サンプル) -> loss=1.0, バッチ2 (1サンプル) -> loss=2.0 の場合,
    サンプル重み付け平均 = (1.0*4 + 2.0*1) / 5 = 1.2
    バッチ平均だと (1.0 + 2.0) / 2 = 1.5 になるため, 区別可能.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        torch.manual_seed(42)
        trainer = PochiTrainer(
            model_name="resnet18",
            num_classes=3,
            pretrained=False,
            device="cpu",
            work_dir=temp_dir,
        )
        trainer.setup_training(learning_rate=0.0, optimizer_name="SGD", num_classes=3)

        data = torch.randn(5, 3, 224, 224)
        targets = torch.tensor([0, 1, 2, 0, 1])
        dataset = TensorDataset(data, targets)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)

        assert trainer.optimizer is not None
        epoch_runner = EpochRunner(device=trainer.device, logger=trainer.logger)

        call_count = 0

        def fake_criterion(output, target):
            nonlocal call_count
            call_count += 1
            # 逆伝播可能な tensor を返す必要がある
            if call_count == 1:
                return torch.tensor(1.0, requires_grad=True)
            return torch.tensor(2.0, requires_grad=True)

        metrics = epoch_runner.run(
            model=trainer.model,
            optimizer=trainer.optimizer,
            criterion=fake_criterion,  # type: ignore[arg-type]
            train_loader=loader,
            epoch=0,
        )

        expected_loss = (1.0 * 4 + 2.0 * 1) / 5
        assert abs(metrics["loss"] - expected_loss) < 1e-6


def test_evaluator_with_empty_loader():
    logger = logging.getLogger("test_evaluator_empty_loader")
    evaluator = Evaluator(device=torch.device("cpu"), logger=logger)
    model = torch.nn.Linear(2, 2)
    criterion = torch.nn.CrossEntropyLoss()

    metrics = evaluator.validate(
        model=model,
        val_loader=_empty_loader(),
        criterion=criterion,
    )

    assert metrics["val_loss"] == 0.0
    assert metrics["val_accuracy"] == 0.0
