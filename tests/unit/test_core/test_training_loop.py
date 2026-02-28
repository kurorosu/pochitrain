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


def test_early_stopping_saves_last_checkpoint_only_once():
    """Early Stopping発動時にlast checkpointが1回だけ保存されることをテスト."""
    checkpoint_store = Mock()
    early_stopping = Mock()
    early_stopping.monitor = "val_accuracy"
    early_stopping.should_stop = False
    early_stopping.step.return_value = True
    training_loop = TrainingLoop(
        logger=Mock(),
        checkpoint_store=checkpoint_store,
        early_stopping=early_stopping,
    )
    model = nn.Linear(2, 2)
    train_loader = cast(DataLoader[Any], Mock())
    val_loader = cast(DataLoader[Any], Mock())

    _, best_accuracy = training_loop.run(
        epochs=3,
        train_epoch_fn=lambda _: {"loss": 1.0, "accuracy": 10.0},
        validate_fn=lambda _: {"val_loss": 1.0, "val_accuracy": 85.0},
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=None,
        scheduler=None,
        tracker=None,
        get_learning_rate_fn=lambda: 0.001,
        get_layer_wise_rates_fn=lambda: {},
        is_layer_wise_lr_fn=lambda: False,
    )

    assert best_accuracy == 85.0
    checkpoint_store.save_best_model.assert_called_once()
    assert checkpoint_store.save_last_model.call_count == 1


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
