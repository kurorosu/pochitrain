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
from .training.evaluator import Evaluator
from .training.metrics_tracker import MetricsTracker
from .training.training_configurator import TrainingConfigurator
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
        self.logger.info(f"使用デバイス: {self.device}")
        if self.cudnn_benchmark:
            self.logger.info("cuDNN自動チューニング: 有効")
        self.logger.info(f"ワークスペース: {self.current_workspace}")
        self.logger.info(f"モデル保存先: {self.work_dir}")

        # チェックポイントストアの初期化
        self.checkpoint_store = CheckpointStore(self.work_dir, self.logger)

        # 検証器の初期化
        self.evaluator = Evaluator(self.device, self.logger)

        # 訓練コンフィギュレータの初期化
        self.training_configurator = TrainingConfigurator(self.device, self.logger)

        # モデルの作成
        self.model = create_model(model_name, num_classes, pretrained)
        self.model.to(self.device)

        # モデル情報の表示
        model_info = self.model.get_model_info()
        self.logger.info(f"モデル: {model_info['model_name']}")
        self.logger.info(f"クラス数: {model_info['num_classes']}")
        self.logger.info(f"総パラメータ数: {model_info['total_params']:,}")
        self.logger.info(f"訓練可能パラメータ数: {model_info['trainable_params']:,}")

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
            self.logger.info(f"混同行列計算を有効化しました (クラス数: {num_classes})")

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

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """1エポックの訓練."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # 勾配をゼロにリセット
            if self.optimizer is not None:
                self.optimizer.zero_grad()

            # 順伝播
            output = self.model(data)
            if self.criterion is not None:
                loss = self.criterion(output, target)
            else:
                raise RuntimeError("criterion is not set")

            # 逆伝播
            loss.backward()
            if self.optimizer is not None:
                self.optimizer.step()

            # 統計情報の更新
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # ログ出力
            if batch_idx % 100 == 0:
                self.logger.info(
                    f"エポック {self.epoch}, バッチ {batch_idx}/{len(train_loader)}, "
                    f"損失: {loss.item():.4f}, 精度: {100.0 * correct / total:.2f}%"
                )

        # エポックの統計情報
        # 例外回避のための防御的ガード. 本来はバリデーションで止めるのが望ましい
        loader_len = len(train_loader)
        avg_loss = total_loss / loader_len if loader_len > 0 else 0.0
        accuracy = 100.0 * correct / total if total > 0 else 0.0

        return {"loss": avg_loss, "accuracy": accuracy}

    def validate(self, val_loader: DataLoader[Any]) -> Dict[str, float]:
        """検証 (Evaluator に委譲)."""
        if self.criterion is None:
            raise RuntimeError("criterion is not set")
        return self.evaluator.validate(
            model=self.model,
            val_loader=val_loader,
            criterion=self.criterion,
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
        self.logger.info(f"訓練を開始 - エポック数: {epochs}")

        # MetricsTrackerの初期化（ワークスペースがある場合のみ）
        tracker = None
        if self.current_workspace is not None:
            tracker = MetricsTracker(
                logger=self.logger,
                visualization_dir=self.workspace_manager.get_visualization_dir(),
                enable_metrics_export=self.enable_metrics_export,
                enable_gradient_tracking=self.enable_gradient_tracking,
                gradient_tracking_config=self.gradient_tracking_config,
                layer_wise_lr_graph_config=self.layer_wise_lr_graph_config,
            )
            tracker.initialize()

        for epoch in range(1, epochs + 1):
            self.epoch = epoch

            # 停止フラグのチェック（エポック開始前）
            if stop_flag_callback and stop_flag_callback():
                self.logger.warning(
                    f"安全停止が要求されました。エポック {epoch-1} で訓練を終了します。"
                )
                self.checkpoint_store.save_last_model(
                    epoch=self.epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    best_accuracy=self.best_accuracy,
                )  # 現在の状態を保存
                break

            self.logger.info(f"エポック {epoch}/{epochs} を開始")

            # 1エポック訓練
            train_metrics = self.train_epoch(train_loader)

            # 検証
            val_metrics = {}
            if val_loader:
                val_metrics = self.validate(val_loader)

            # スケジューラーの更新
            if self.scheduler is not None:
                self.scheduler.step()

            # ログ出力
            self.logger.info(
                f"エポック {epoch} 完了 - "
                f"訓練損失: {train_metrics['loss']:.4f}, "
                f"訓練精度: {train_metrics['accuracy']:.2f}%"
            )

            if val_metrics:
                self.logger.info(
                    f"検証損失: {val_metrics['val_loss']:.4f}, "
                    f"検証精度: {val_metrics['val_accuracy']:.2f}%"
                )

                # ベストモデルの更新（精度が前回以上なら保存）
                if val_metrics["val_accuracy"] >= self.best_accuracy:
                    self.best_accuracy = val_metrics["val_accuracy"]
                    self.checkpoint_store.save_best_model(
                        epoch=epoch,
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        best_accuracy=self.best_accuracy,
                    )

                # Early Stopping判定
                if self.early_stopping is not None:
                    monitor = self.early_stopping.monitor
                    monitor_value = val_metrics.get(monitor)
                    if monitor_value is not None:
                        if self.early_stopping.step(monitor_value, epoch):
                            self.checkpoint_store.save_last_model(
                                epoch=self.epoch,
                                model=self.model,
                                optimizer=self.optimizer,
                                scheduler=self.scheduler,
                                best_accuracy=self.best_accuracy,
                            )
                            break

            # ラストモデルの保存（毎エポック上書き）
            self.checkpoint_store.save_last_model(
                epoch=self.epoch,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                best_accuracy=self.best_accuracy,
            )

            # メトリクスと勾配の記録
            if tracker is not None:
                layer_wise_rates = {}
                if self.is_layer_wise_lr_enabled():
                    layer_rates = self.get_layer_wise_learning_rates()
                    for layer_name, lr in layer_rates.items():
                        layer_wise_rates[f"lr_{layer_name}"] = lr

                tracker.record_epoch(
                    epoch=epoch,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    model=self.model,
                    learning_rate=self._get_base_learning_rate(),
                    layer_wise_lr_enabled=self.is_layer_wise_lr_enabled(),
                    layer_wise_rates=layer_wise_rates,
                )

            # 停止フラグのチェック（エポック完了後）
            if stop_flag_callback and stop_flag_callback():
                self.logger.warning(
                    f"安全停止が要求されました。エポック {epoch} で訓練を終了します。"
                )
                break

        # Early Stoppingによる停止の場合、その旨をログ出力
        if self.early_stopping is not None and self.early_stopping.should_stop:
            self.logger.info(
                f"Early Stoppingにより訓練を停止しました "
                f"(最終エポック: {self.epoch}, "
                f"ベスト{self.early_stopping.monitor}: "
                f"{self.early_stopping.best_value:.4f}, "
                f"ベストエポック: {self.early_stopping.best_epoch})"
            )
        else:
            self.logger.info("訓練が完了しました")
        if val_loader:
            self.logger.info(f"最高精度: {self.best_accuracy:.2f}%")

        # 訓練完了後のエクスポート処理
        if tracker is not None:
            csv_path, graph_paths = tracker.finalize()
            if csv_path:
                self.logger.info(f"メトリクスCSVを出力: {csv_path}")
            if graph_paths:
                for graph_path in graph_paths:
                    self.logger.info(f"メトリクスグラフを出力: {graph_path}")

            # サマリー情報の表示
            summary = tracker.get_summary()
            if summary:
                self.logger.info("=== 訓練サマリー ===")
                self.logger.info(f"総エポック数: {summary['total_epochs']}")
                self.logger.info(f"最終訓練損失: {summary['final_train_loss']:.4f}")
                self.logger.info(
                    f"最終訓練精度: {summary['final_train_accuracy']:.2f}%"
                )
                if "best_val_accuracy" in summary:
                    self.logger.info(
                        f"最高検証精度: {summary['best_val_accuracy']:.2f}% "
                        f"(エポック {summary['best_val_accuracy_epoch']})"
                    )

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
