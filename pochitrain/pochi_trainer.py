"""
pochitrain.pochi_trainer: Pochiトレーナー.

複雑なレジストリシステムを使わない、直接的なトレーナー
"""

import dataclasses
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .config import PochiConfig
from .logging import LoggerManager
from .models.pochi_models import create_model
from .training.checkpoint_store import CheckpointStore
from .training.early_stopping import EarlyStopping
from .training.epoch_runner import EpochRunner
from .training.evaluator import Evaluator
from .training.training_configurator import TrainingConfigurator
from .training.training_loop import TrainingLoop
from .utils.directory_manager import PochiWorkspaceManager


class PochiTrainer:
    """
    Pochiトレーナークラス.

    Args:
        model_name (str): モデル名 ('resnet18', 'resnet34', 'resnet50')
        num_classes (int): 分類クラス数
        pretrained (bool): 事前学習済みモデルを使用するか
        device (str): デバイス ('cuda' or 'cpu') - 必須設定
        work_dir (str, optional): 作業ディレクトリ
        create_workspace (bool, optional): ワークスペースを作成するか（推論時はFalse）
    """

    # 型アノテーション（推論時にワークスペース作成をスキップするため、current_workspaceがNoneになる可能性がある）
    current_workspace: Optional[
        Path
    ]  # 推論モードではNone、訓練モードではPathオブジェクト
    work_dir: Path  # ワークスペース作成の有無に関わらず常にPathオブジェクト

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        device: str,
        pretrained: bool = True,
        work_dir: str = "work_dirs",
        create_workspace: bool = True,
        cudnn_benchmark: bool = False,
    ):
        """PochiTrainerを初期化."""
        # モデル設定の保存
        self.model_name = model_name
        self.num_classes = num_classes

        # デバイスの設定（バリデーション済みの設定を使用）
        self.device = torch.device(device)

        # cuDNN自動チューニングの設定
        if self.device.type == "cuda" and cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
            self.cudnn_benchmark = True
        else:
            self.cudnn_benchmark = False

        # ワークスペースマネージャーの初期化
        self.workspace_manager = PochiWorkspaceManager(work_dir)
        if create_workspace:
            self.current_workspace = self.workspace_manager.create_workspace()
            self.work_dir = self.workspace_manager.get_models_dir()
        else:
            self.current_workspace = None
            self.work_dir = Path(work_dir)

        # ロガーの設定
        self.logger = self._setup_logger()
        self.logger.debug(f"使用デバイス: {self.device}")
        if self.cudnn_benchmark:
            self.logger.debug("cuDNN自動チューニング: 有効")
        self.logger.debug(f"ワークスペース: {self.current_workspace}")
        self.logger.debug(f"モデル保存先: {self.work_dir}")

        # チェックポイントストアの初期化
        self.checkpoint_store = CheckpointStore(self.work_dir, self.logger)

        # 検証器の初期化
        self.evaluator = Evaluator(self.device, self.logger)

        # 訓練コンフィギュレータの初期化
        self.training_configurator = TrainingConfigurator(self.device, self.logger)
        self.epoch_runner = EpochRunner(device=self.device, logger=self.logger)

        # モデルの作成
        self.model = create_model(model_name, num_classes, pretrained)
        self.model.to(self.device)

        # モデル情報の表示
        model_info = self.model.get_model_info()
        self.logger.debug(f"モデル: {model_info['model_name']}")
        self.logger.debug(f"クラス数: {model_info['num_classes']}")
        self.logger.debug(f"総パラメータ数: {model_info['total_params']:,}")
        self.logger.debug(f"訓練可能パラメータ数: {model_info['trainable_params']:,}")

        # 訓練状態の管理
        self.epoch = 0
        self.best_accuracy = 0.0

        # 混同行列計算のためのクラス数（後で設定）
        self.num_classes_for_cm: Optional[int] = None

        # 最適化器・損失関数は後で設定
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler.LRScheduler] = None
        self.criterion: Optional[nn.Module] = None

        # メトリクス・勾配の設定（CLIから設定される）
        self.enable_metrics_export = True  # デフォルトで有効
        self.enable_gradient_tracking = False  # デフォルトでOFF（計算コスト考慮）
        self.gradient_tracking_config: Dict[str, Any] = {
            "record_frequency": 1,  # 記録頻度（1 = 毎エポック）
        }

        # Early Stopping（setup_training()で初期化）
        self.early_stopping: Optional[EarlyStopping] = None

    @classmethod
    def from_config(
        cls, config: PochiConfig, create_workspace: bool = True
    ) -> "PochiTrainer":
        """PochiConfigからトレーナーを作成."""
        return cls(
            model_name=config.model_name,
            num_classes=config.num_classes,
            device=config.device,
            pretrained=config.pretrained,
            work_dir=config.work_dir,
            create_workspace=create_workspace,
            cudnn_benchmark=config.cudnn_benchmark,
        )

    def setup_training_from_config(self, config: PochiConfig, num_classes: int) -> None:
        """PochiConfigから訓練設定を適用."""
        layer_wise_lr_config = dataclasses.asdict(config.layer_wise_lr_config)
        early_stopping_config = (
            dataclasses.asdict(config.early_stopping)
            if config.early_stopping is not None
            else None
        )
        self.setup_training(
            learning_rate=config.learning_rate,
            optimizer_name=config.optimizer,
            scheduler_name=config.scheduler,
            scheduler_params=config.scheduler_params,
            class_weights=config.class_weights,
            num_classes=num_classes,
            enable_layer_wise_lr=config.enable_layer_wise_lr,
            layer_wise_lr_config=layer_wise_lr_config,
            early_stopping_config=early_stopping_config,
        )

    def _setup_logger(self) -> logging.Logger:
        """ロガーの設定."""
        logger_manager = LoggerManager()
        return logger_manager.get_logger("pochitrain")

    def setup_training(
        self,
        learning_rate: float = 0.001,
        optimizer_name: str = "Adam",
        scheduler_name: Optional[str] = None,
        scheduler_params: Optional[dict] = None,
        class_weights: Optional[List[float]] = None,
        num_classes: Optional[int] = None,
        enable_layer_wise_lr: bool = False,
        layer_wise_lr_config: Optional[Dict[str, Any]] = None,
        early_stopping_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        訓練の設定.

        Args:
            learning_rate (float): 学習率
            optimizer_name (str): 最適化器名 ('Adam', 'AdamW', 'SGD')
            scheduler_name (str, optional): スケジューラー名
                ('StepLR', 'MultiStepLR', 'CosineAnnealingLR',
                'ExponentialLR', 'LinearLR')
            scheduler_params (dict, optional): スケジューラーのパラメータ
            class_weights (List[float], optional): クラス毎の損失重み
            num_classes (int, optional): クラス数（重みのバリデーション用）
            enable_layer_wise_lr (bool): 層別学習率を有効にするか
            layer_wise_lr_config (Dict[str, Any], optional): 層別学習率の設定
            early_stopping_config (Dict[str, Any], optional): Early Stopping設定
        """
        components = self.training_configurator.configure(
            model=self.model,
            learning_rate=learning_rate,
            optimizer_name=optimizer_name,
            scheduler_name=scheduler_name,
            scheduler_params=scheduler_params,
            class_weights=class_weights,
            num_classes=num_classes,
            enable_layer_wise_lr=enable_layer_wise_lr,
            layer_wise_lr_config=layer_wise_lr_config,
        )

        self.optimizer = components.optimizer
        self.scheduler = components.scheduler
        self.criterion = components.criterion
        self.enable_layer_wise_lr = components.enable_layer_wise_lr
        self.base_learning_rate = components.base_learning_rate
        self.layer_wise_lr_config = components.layer_wise_lr_config
        self.layer_wise_lr_graph_config = components.layer_wise_lr_graph_config

        # 混同行列計算のためのクラス数を設定
        if num_classes:
            self.num_classes_for_cm = num_classes
            self.logger.debug(f"混同行列計算を有効化しました (クラス数: {num_classes})")

        # Early Stoppingの初期化
        if early_stopping_config is not None and early_stopping_config.get(
            "enabled", False
        ):
            self.early_stopping = EarlyStopping(
                patience=early_stopping_config.get("patience", 10),
                min_delta=early_stopping_config.get("min_delta", 0.0),
                monitor=early_stopping_config.get("monitor", "val_accuracy"),
                logger=self.logger,
            )
            self.logger.info(
                f"Early Stopping: 有効 "
                f"(patience={self.early_stopping.patience}, "
                f"min_delta={self.early_stopping.min_delta}, "
                f"monitor={self.early_stopping.monitor})"
            )

    def _ensure_training_configured(self) -> None:
        """訓練に必要なコンポーネントが設定済みか検証.

        Raises:
            RuntimeError: setup_training() が未実行の場合.
        """
        if self.optimizer is None or self.criterion is None:
            raise RuntimeError(
                "訓練コンポーネントが未設定です. "
                "train() の前に setup_training() を呼び出してください."
            )

    def _require_training_components(self) -> tuple[optim.Optimizer, nn.Module]:
        """訓練に必要なコンポーネントを取得.

        Returns:
            tuple[optim.Optimizer, nn.Module]: (optimizer, criterion).

        Raises:
            RuntimeError: setup_training() が未実行の場合.
        """
        self._ensure_training_configured()
        if self.optimizer is None or self.criterion is None:
            # mypy向けの型保証. 実運用上は _ensure_training_configured で到達しない.
            raise RuntimeError(
                "訓練コンポーネントが未設定です. "
                "train() の前に setup_training() を呼び出してください."
            )
        return self.optimizer, self.criterion

    def _require_criterion(self) -> nn.Module:
        """損失関数を取得.

        Returns:
            nn.Module: criterion.

        Raises:
            RuntimeError: criterion が未設定の場合.
        """
        if self.criterion is None:
            raise RuntimeError("criterion is not set")
        return self.criterion

    def _set_epoch(self, epoch: int) -> None:
        """現在エポックを更新."""
        self.epoch = epoch

    def _run_train_epoch(self, train_loader: DataLoader[Any]) -> Dict[str, float]:
        """1エポック訓練を実行."""
        optimizer, criterion = self._require_training_components()
        return self.epoch_runner.run(
            model=self.model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            epoch=self.epoch,
        )

    def train_one_epoch(
        self,
        epoch: int,
        train_loader: DataLoader[Any],
    ) -> Dict[str, float]:
        """指定エポックで1エポック訓練を実行.

        Args:
            epoch: 実行するエポック番号.
            train_loader: 訓練データローダー.

        Returns:
            Dict[str, float]: 訓練メトリクス.
        """
        self._set_epoch(epoch)
        return self._run_train_epoch(train_loader)

    def validate(self, val_loader: DataLoader[Any]) -> Dict[str, float]:
        """検証 (Evaluator に委譲)."""
        criterion = self._require_criterion()
        return self.evaluator.validate(
            model=self.model,
            val_loader=val_loader,
            criterion=criterion,
            num_classes_for_cm=self.num_classes_for_cm,
            epoch=self.epoch,
            workspace_path=self.current_workspace,
        )

    def train(
        self,
        train_loader: DataLoader[Any],
        val_loader: Optional[DataLoader[Any]] = None,
        epochs: int = 10,
        stop_flag_callback: Optional[Any] = None,
    ) -> None:
        """
        訓練の実行.

        Args:
            train_loader (DataLoader): 訓練データローダー
            val_loader (DataLoader, optional): 検証データローダー
            epochs (int): エポック数
            stop_flag_callback (callable, optional): 停止フラグをチェックするコールバック関数
        """
        self._ensure_training_configured()
        self.logger.debug(f"訓練を開始 - エポック数: {epochs}")

        training_loop = TrainingLoop(
            logger=self.logger,
            checkpoint_store=self.checkpoint_store,
            early_stopping=self.early_stopping,
        )

        visualization_dir = (
            self.workspace_manager.get_visualization_dir()
            if self.current_workspace is not None
            else None
        )
        tracker = TrainingLoop.create_metrics_tracker(
            logger=self.logger,
            current_workspace=self.current_workspace,
            visualization_dir=visualization_dir,
            enable_metrics_export=self.enable_metrics_export,
            enable_gradient_tracking=self.enable_gradient_tracking,
            gradient_tracking_config=self.gradient_tracking_config,
            layer_wise_lr_graph_config=self.layer_wise_lr_graph_config,
        )

        last_epoch, best_accuracy = training_loop.run(
            epochs=epochs,
            train_epoch_fn=self._run_train_epoch,
            validate_fn=self.validate,
            train_loader=train_loader,
            val_loader=val_loader,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            tracker=tracker,
            get_learning_rate_fn=self._get_base_learning_rate,
            get_layer_wise_rates_fn=self.get_layer_wise_learning_rates,
            is_layer_wise_lr_fn=self.is_layer_wise_lr_enabled,
            initial_best_accuracy=self.best_accuracy,
            set_epoch_fn=self._set_epoch,
            stop_flag_callback=stop_flag_callback,
        )

        self.epoch = last_epoch
        self.best_accuracy = best_accuracy

    def _get_base_learning_rate(self) -> float:
        """現在の基本学習率を取得.

        Returns:
            float: 基本学習率 (層別学習率有効時は設定値, 無効時は実際の学習率).
        """
        if self.enable_layer_wise_lr:
            return self.base_learning_rate
        else:
            if self.optimizer is not None:
                lr: float = self.optimizer.param_groups[0]["lr"]
                return lr
            return 0.0

    def is_layer_wise_lr_enabled(self) -> bool:
        """層別学習率が有効かどうかを判定.

        Returns:
            bool: 層別学習率が有効な場合True.
        """
        return self.enable_layer_wise_lr

    def get_layer_wise_learning_rates(self) -> Dict[str, float]:
        """各層の現在の学習率を取得.

        Returns:
            Dict[str, float]: 層名をキー, 学習率を値とする辞書.
        """
        layer_rates = {}
        if self.enable_layer_wise_lr and self.optimizer:
            for group in self.optimizer.param_groups:
                layer_name = group.get("layer_name", "unknown")
                layer_rates[layer_name] = group["lr"]
        return layer_rates
