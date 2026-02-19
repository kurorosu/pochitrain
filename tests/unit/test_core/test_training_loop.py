"""TrainingLoopのテスト."""

import logging
from typing import Any, cast
from unittest.mock import Mock

import torch.nn as nn
from torch.utils.data import DataLoader

from pochitrain.training.training_loop import TrainingLoop


def test_create_metrics_tracker_returns_none_without_workspace():
    """workspace未作成時にMetricsTrackerを作成しないことをテスト."""
    tracker = TrainingLoop.create_metrics_tracker(
        logger=logging.getLogger("test"),
        current_workspace=None,
        visualization_dir=None,
        enable_metrics_export=True,
        enable_gradient_tracking=False,
        gradient_tracking_config={},
        layer_wise_lr_graph_config={},
    )

    assert tracker is None


def test_run_uses_initial_best_accuracy():
    """初期ベスト精度を引き継いでbest更新判定することをテスト."""
    checkpoint_store = Mock()
    logger = Mock()
    training_loop = TrainingLoop(
        logger=logger,
        checkpoint_store=checkpoint_store,
        early_stopping=None,
    )
    model = nn.Linear(2, 2)
    train_loader = cast(DataLoader[Any], Mock())
    val_loader = cast(DataLoader[Any], Mock())

    _, best_accuracy = training_loop.run(
        epochs=1,
        train_epoch_fn=lambda _: {"loss": 1.0, "accuracy": 10.0},
        validate_fn=lambda _: {"val_loss": 1.0, "val_accuracy": 80.0},
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=None,
        scheduler=None,
        tracker=None,
        get_learning_rate_fn=lambda: 0.001,
        get_layer_wise_rates_fn=lambda: {},
        is_layer_wise_lr_fn=lambda: False,
        initial_best_accuracy=90.0,
    )

    assert best_accuracy == 90.0
    checkpoint_store.save_best_model.assert_not_called()
    checkpoint_store.save_last_model.assert_called_once()


def test_update_best_and_check_early_stop_does_not_save_last_checkpoint():
    """Early Stopping判定でlast checkpointを重複保存しないことをテスト."""
    checkpoint_store = Mock()
    early_stopping = Mock()
    early_stopping.monitor = "val_accuracy"
    early_stopping.step.return_value = True
    training_loop = TrainingLoop(
        logger=Mock(),
        checkpoint_store=checkpoint_store,
        early_stopping=early_stopping,
    )
    model = nn.Linear(2, 2)

    _, should_stop = training_loop._update_best_and_check_early_stop(
        epoch=1,
        val_metrics={"val_loss": 1.0, "val_accuracy": 85.0},
        model=model,
        optimizer=None,
        scheduler=None,
        best_accuracy=80.0,
    )

    assert should_stop is True
    checkpoint_store.save_best_model.assert_called_once()
    checkpoint_store.save_last_model.assert_not_called()


def test_run_calls_set_epoch_fn_each_epoch():
    """run() が各エポックで set_epoch_fn を呼ぶことをテスト."""
    checkpoint_store = Mock()
    training_loop = TrainingLoop(
        logger=Mock(),
        checkpoint_store=checkpoint_store,
        early_stopping=None,
    )
    model = nn.Linear(2, 2)
    train_loader = cast(DataLoader[Any], Mock())
    val_loader = cast(DataLoader[Any], Mock())
    epochs: list[int] = []

    training_loop.run(
        epochs=2,
        train_epoch_fn=lambda _: {"loss": 1.0, "accuracy": 10.0},
        validate_fn=lambda _: {"val_loss": 1.0, "val_accuracy": 80.0},
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=None,
        scheduler=None,
        tracker=None,
        get_learning_rate_fn=lambda: 0.001,
        get_layer_wise_rates_fn=lambda: {},
        is_layer_wise_lr_fn=lambda: False,
        set_epoch_fn=epochs.append,
    )

    assert epochs == [1, 2]
