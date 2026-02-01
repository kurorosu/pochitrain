"""pochitrain.training.training_configurator: optimizer/scheduler/criterion の構築."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class TrainingComponents:
    """configure() の戻り値."""

    optimizer: optim.Optimizer
    scheduler: Optional[optim.lr_scheduler.LRScheduler]
    criterion: nn.Module
    enable_layer_wise_lr: bool
    base_learning_rate: float
    layer_wise_lr_config: Dict[str, Any] = field(default_factory=dict)
    layer_wise_lr_graph_config: Dict[str, Any] = field(default_factory=dict)


class TrainingConfigurator:
    """optimizer, scheduler, criterion の構築を担当."""

    def __init__(self, device: torch.device, logger: logging.Logger):
        """訓練コンフィギュレータを初期化.

        Args:
            device: 訓練に使用するデバイス.
            logger: ロガー.
        """
        self.device = device
        self.logger = logger

    def configure(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        optimizer_name: str = "Adam",
        scheduler_name: Optional[str] = None,
        scheduler_params: Optional[dict] = None,
        class_weights: Optional[List[float]] = None,
        num_classes: Optional[int] = None,
        enable_layer_wise_lr: bool = False,
        layer_wise_lr_config: Optional[Dict[str, Any]] = None,
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

        # 損失関数の設定 (クラス重み対応)
        criterion = self._build_criterion(class_weights, num_classes)

        # パラメータグループの構築
        if enable_layer_wise_lr:
            param_groups = self._build_layer_wise_param_groups(
                model, learning_rate, lr_config
            )
            self._log_layer_wise_lr(param_groups)
        else:
            param_groups = [{"params": model.parameters(), "lr": learning_rate}]

        # 最適化器の構築
        optimizer = self._build_optimizer(optimizer_name, param_groups)

        # スケジューラーの構築
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
        class_weights: Optional[List[float]],
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
        param_groups: List[Dict[str, Any]],
    ) -> optim.Optimizer:
        """最適化器を構築.

        Args:
            optimizer_name: 最適化器名.
            param_groups: パラメータグループ.

        Returns:
            optim.Optimizer: 最適化器.
        """
        if optimizer_name == "Adam":
            return optim.Adam(param_groups)
        elif optimizer_name == "AdamW":
            return optim.AdamW(param_groups, weight_decay=1e-2)
        elif optimizer_name == "SGD":
            return optim.SGD(param_groups, momentum=0.9, weight_decay=1e-4)
        else:
            raise ValueError(f"サポートされていない最適化器: {optimizer_name}")

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

        schedulers: Dict[str, type] = {
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

    def _build_layer_wise_param_groups(
        self,
        model: nn.Module,
        base_lr: float,
        layer_wise_lr_config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """層別学習率のパラメータグループを構築.

        Args:
            model: 訓練対象のモデル.
            base_lr: 基本学習率.
            layer_wise_lr_config: 層別学習率の設定.

        Returns:
            List[Dict[str, Any]]: パラメータグループのリスト.
        """
        layer_rates = layer_wise_lr_config.get("layer_rates", {})

        # 層ごとにパラメータを分類
        layer_params: Dict[str, List] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                layer_group = self._get_layer_group(name)
                if layer_group not in layer_params:
                    layer_params[layer_group] = []
                layer_params[layer_group].append(param)

        # パラメータグループを作成
        param_groups = []
        for layer_name, params in layer_params.items():
            lr = layer_rates.get(layer_name, base_lr)
            param_groups.append(
                {
                    "params": params,
                    "lr": lr,
                    "layer_name": layer_name,
                }
            )

        return param_groups

    def _get_layer_group(self, param_name: str) -> str:
        """パラメータ名から層グループ名を取得.

        Args:
            param_name: パラメータ名.

        Returns:
            str: 層グループ名.
        """
        # ResNetの構造に基づいて層を分類 (順序重要: より具体的なものから先に判定)
        if "layer1" in param_name:
            return "layer1"
        elif "layer2" in param_name:
            return "layer2"
        elif "layer3" in param_name:
            return "layer3"
        elif "layer4" in param_name:
            return "layer4"
        elif "conv1" in param_name:
            return "conv1"
        elif "bn1" in param_name:
            return "bn1"
        elif "fc" in param_name:
            return "fc"
        else:
            return "other"

    def _log_layer_wise_lr(self, param_groups: List[Dict[str, Any]]) -> None:
        """層別学習率の設定をログ出力.

        Args:
            param_groups: パラメータグループのリスト.
        """
        self.logger.debug("=== 層別学習率設定 ===")
        for group in param_groups:
            layer_name = group.get("layer_name", "unknown")
            lr = group["lr"]
            param_count = sum(p.numel() for p in group["params"])
            self.logger.debug(f"  {layer_name}: lr={lr:.6f}, params={param_count:,}")
        self.logger.debug("=====================")
