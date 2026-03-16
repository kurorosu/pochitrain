"""pochitrain.training.training_configurator: optimizer/scheduler/criterion の構築."""

import logging
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from .layer_wise_lr import ParamGroupBuilder


@dataclass
class TrainingComponents:
    """configure() の戻り値."""

    optimizer: optim.Optimizer
    scheduler: Optional[optim.lr_scheduler.LRScheduler]
    criterion: nn.Module
    enable_layer_wise_lr: bool
    base_learning_rate: float
    layer_wise_lr_config: dict[str, Any] = field(default_factory=dict)
    layer_wise_lr_graph_config: dict[str, Any] = field(default_factory=dict)


class TrainingConfigurator:
    """optimizer, scheduler, criterion の構築を担当."""

    def __init__(
        self,
        device: torch.device,
        logger: logging.Logger,
        param_group_builder: Optional[ParamGroupBuilder] = None,
    ):
        """訓練コンフィギュレータを初期化.

        Args:
            device: 訓練に使用するデバイス.
            logger: ロガー.
            param_group_builder: 層別学習率のパラメータグループビルダー.
        """
        self.device = device
        self.logger = logger
        self._param_group_builder = param_group_builder

    def configure(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        optimizer_name: str = "Adam",
        scheduler_name: Optional[str] = None,
        scheduler_params: Optional[dict] = None,
        class_weights: Optional[list[float]] = None,
        num_classes: Optional[int] = None,
        enable_layer_wise_lr: bool = False,
        layer_wise_lr_config: Optional[dict[str, Any]] = None,
    ) -> TrainingComponents:
        """optimizer, scheduler, criterion を構築.

        Args:
            model: 訓練対象のモデル.
            learning_rate: 学習率.
            optimizer_name: 最適化器名 ('Adam', 'AdamW', 'SGD').
            scheduler_name: スケジューラー名.
            scheduler_params: スケジューラーのパラメータ.
            class_weights: クラス毎の損失重み.
            num_classes: クラス数 (重みのバリデーション用).
            enable_layer_wise_lr: 層別学習率を有効にするか.
            layer_wise_lr_config: 層別学習率の設定.

        Returns:
            TrainingComponents: 構築された訓練コンポーネント.
        """
        lr_config = layer_wise_lr_config or {}

        criterion = self._build_criterion(class_weights, num_classes)

        if enable_layer_wise_lr:
            if self._param_group_builder is None:
                raise RuntimeError(
                    "層別学習率を使用するには param_group_builder の注入が必要です"
                )
            param_groups = self._param_group_builder.build(
                model, learning_rate, lr_config
            )
            self._param_group_builder.log_param_groups(param_groups)
        else:
            param_groups = [{"params": model.parameters(), "lr": learning_rate}]

        optimizer = self._build_optimizer(optimizer_name, param_groups)

        scheduler = self._build_scheduler(optimizer, scheduler_name, scheduler_params)

        self.logger.debug(f"最適化器: {optimizer_name} (学習率: {learning_rate})")
        if scheduler_name:
            self.logger.debug(f"スケジューラー: {scheduler_name}")

        return TrainingComponents(
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            enable_layer_wise_lr=enable_layer_wise_lr,
            base_learning_rate=learning_rate,
            layer_wise_lr_config=lr_config,
            layer_wise_lr_graph_config=lr_config.get("graph_config", {}),
        )

    def _build_criterion(
        self,
        class_weights: Optional[list[float]],
        num_classes: Optional[int],
    ) -> nn.Module:
        """損失関数を構築.

        Args:
            class_weights: クラス毎の損失重み.
            num_classes: クラス数 (重みのバリデーション用).

        Returns:
            nn.Module: 損失関数.
        """
        if class_weights is not None:
            if num_classes is not None and len(class_weights) != num_classes:
                raise ValueError(
                    f"クラス重みの長さ({len(class_weights)})が"
                    f"クラス数({num_classes})と一致しません"
                )
            weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(
                self.device
            )
            self.logger.debug(f"クラス重みを設定: {class_weights}")
            return nn.CrossEntropyLoss(weight=weights_tensor)
        return nn.CrossEntropyLoss()

    def _build_optimizer(
        self,
        optimizer_name: str,
        param_groups: list[dict[str, Any]],
    ) -> optim.Optimizer:
        """最適化器を構築.

        Args:
            optimizer_name: 最適化器名.
            param_groups: パラメータグループ.

        Returns:
            optim.Optimizer: 最適化器.
        """
        optimizers: dict[str, Any] = {
            "Adam": optim.Adam,
            "AdamW": partial(optim.AdamW, weight_decay=1e-2),
            "SGD": partial(optim.SGD, momentum=0.9, weight_decay=1e-4),
        }

        optimizer_cls = optimizers.get(optimizer_name)
        if optimizer_cls is None:
            raise ValueError(f"サポートされていない最適化器: {optimizer_name}")
        optimizer: optim.Optimizer = optimizer_cls(param_groups)
        return optimizer

    def _build_scheduler(
        self,
        optimizer: optim.Optimizer,
        scheduler_name: Optional[str],
        scheduler_params: Optional[dict],
    ) -> Optional[optim.lr_scheduler.LRScheduler]:
        """スケジューラーを構築.

        Args:
            optimizer: 最適化器.
            scheduler_name: スケジューラー名.
            scheduler_params: スケジューラーのパラメータ.

        Returns:
            Optional[optim.lr_scheduler.LRScheduler]: スケジューラー.
        """
        if scheduler_name is None:
            return None

        if scheduler_params is None:
            raise ValueError(
                f"スケジューラー '{scheduler_name}' を使用する場合、"
                f"scheduler_paramsが必須です。configs/pochi_config.pyで設定してください。"
            )

        schedulers: dict[str, type] = {
            "StepLR": optim.lr_scheduler.StepLR,
            "MultiStepLR": optim.lr_scheduler.MultiStepLR,
            "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR,
            "ExponentialLR": optim.lr_scheduler.ExponentialLR,
            "LinearLR": optim.lr_scheduler.LinearLR,
        }

        scheduler_cls = schedulers.get(scheduler_name)
        if scheduler_cls is None:
            raise ValueError(f"サポートされていないスケジューラー: {scheduler_name}")
        scheduler: optim.lr_scheduler.LRScheduler = scheduler_cls(
            optimizer, **scheduler_params
        )
        return scheduler
