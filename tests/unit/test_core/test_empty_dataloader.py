import logging
import tempfile

import torch
from torch.utils.data import DataLoader, TensorDataset

from pochitrain.pochi_trainer import PochiTrainer
from pochitrain.training.evaluator import Evaluator


def _empty_loader(batch_size: int = 4) -> DataLoader:
    data = torch.empty((0, 3, 224, 224))
    targets = torch.empty((0,), dtype=torch.long)
    dataset = TensorDataset(data, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def test_train_epoch_with_empty_loader():
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

        metrics = trainer.train_epoch(_empty_loader())

        assert metrics["loss"] == 0.0
        assert metrics["accuracy"] == 0.0


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
