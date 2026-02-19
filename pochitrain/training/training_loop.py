"""pochitrain.training.training_loop: エポックサイクルの実行を管理するモジュール."""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .checkpoint_store import CheckpointStore
from .early_stopping import EarlyStopping
from .metrics_tracker import MetricsTracker


class TrainingLoop:
    """訓練ループの実行を管理するクラス.

    エポックサイクル（訓練・検証・チェックポイント保存・メトリクス記録）を
    一元管理し, PochiTrainer から訓練ループの詳細を分離する.

    Args:
        logger: ロガーインスタンス.
        checkpoint_store: チェックポイント保存を管理するストア.
        early_stopping: Early Stopping設定. 無効の場合はNone.
    """

    def __init__(
        self,
        logger: logging.Logger,
        checkpoint_store: CheckpointStore,
        early_stopping: Optional[EarlyStopping],
    ) -> None:
        """TrainingLoopを初期化."""
        self.logger = logger
        self.checkpoint_store = checkpoint_store
        self.early_stopping = early_stopping

    def run(
        self,
        epochs: int,
        train_epoch_fn: Callable[[DataLoader[Any]], Dict[str, float]],
        validate_fn: Callable[[DataLoader[Any]], Dict[str, float]],
        train_loader: DataLoader[Any],
        val_loader: Optional[DataLoader[Any]],
        model: nn.Module,
        optimizer: Optional[optim.Optimizer],
        scheduler: Optional[optim.lr_scheduler.LRScheduler],
        tracker: Optional[MetricsTracker],
        get_learning_rate_fn: Callable[[], float],
        get_layer_wise_rates_fn: Callable[[], Dict[str, float]],
        is_layer_wise_lr_fn: Callable[[], bool],
        initial_best_accuracy: float = 0.0,
        set_epoch_fn: Optional[Callable[[int], None]] = None,
        stop_flag_callback: Optional[Any] = None,
    ) -> tuple[int, float]:
        """訓練ループを実行.

        Args:
            epochs: 総エポック数.
            train_epoch_fn: 1エポック訓練を実行する関数.
            validate_fn: 検証を実行する関数.
            train_loader: 訓練データローダー.
            val_loader: 検証データローダー.
            model: モデル.
            optimizer: オプティマイザ.
            scheduler: スケジューラ.
            tracker: メトリクストラッカー.
            get_learning_rate_fn: 現在の学習率を取得する関数.
            get_layer_wise_rates_fn: 層別学習率を取得する関数.
            is_layer_wise_lr_fn: 層別学習率が有効か判定する関数.
            initial_best_accuracy: 学習開始時点のベスト精度.
            set_epoch_fn: 現在エポックを更新する関数.
            stop_flag_callback: 停止フラグをチェックするコールバック関数.

        Returns:
            tuple[int, float]: (最終エポック番号, ベスト精度).
        """
        best_accuracy = initial_best_accuracy
        last_epoch = 0

        for epoch in range(1, epochs + 1):
            last_epoch = epoch
            if set_epoch_fn is not None:
                set_epoch_fn(epoch)

            # 停止フラグのチェック（エポック開始前）
            if stop_flag_callback and stop_flag_callback():
                self.logger.warning(
                    f"安全停止が要求されました。エポック {epoch-1} で訓練を終了します。"
                )
                self._save_last_checkpoint(
                    epoch, model, optimizer, scheduler, best_accuracy
                )
                break

            self.logger.debug(f"エポック {epoch}/{epochs} を開始")

            # 1エポックの訓練・検証・更新サイクル
            best_accuracy, should_stop = self._run_epoch_cycle(
                epoch=epoch,
                train_epoch_fn=train_epoch_fn,
                validate_fn=validate_fn,
                train_loader=train_loader,
                val_loader=val_loader,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                tracker=tracker,
                best_accuracy=best_accuracy,
                get_learning_rate_fn=get_learning_rate_fn,
                get_layer_wise_rates_fn=get_layer_wise_rates_fn,
                is_layer_wise_lr_fn=is_layer_wise_lr_fn,
            )
            if should_stop:
                break

            # 停止フラグのチェック（エポック完了後）
            if stop_flag_callback and stop_flag_callback():
                self.logger.warning(
                    f"安全停止が要求されました。エポック {epoch} で訓練を終了します。"
                )
                break

        self._log_training_result(val_loader, last_epoch, best_accuracy)
        self._finalize_metrics(tracker)

        return last_epoch, best_accuracy

    def _run_epoch_cycle(
        self,
        epoch: int,
        train_epoch_fn: Callable[[DataLoader[Any]], Dict[str, float]],
        validate_fn: Callable[[DataLoader[Any]], Dict[str, float]],
        train_loader: DataLoader[Any],
        val_loader: Optional[DataLoader[Any]],
        model: nn.Module,
        optimizer: Optional[optim.Optimizer],
        scheduler: Optional[optim.lr_scheduler.LRScheduler],
        tracker: Optional[MetricsTracker],
        best_accuracy: float,
        get_learning_rate_fn: Callable[[], float],
        get_layer_wise_rates_fn: Callable[[], Dict[str, float]],
        is_layer_wise_lr_fn: Callable[[], bool],
    ) -> tuple[float, bool]:
        """1エポックの訓練・検証・記録サイクルを実行.

        Args:
            epoch: 現在のエポック番号.
            train_epoch_fn: 1エポック訓練を実行する関数.
            validate_fn: 検証を実行する関数.
            train_loader: 訓練データローダー.
            val_loader: 検証データローダー.
            model: モデル.
            optimizer: オプティマイザ.
            scheduler: スケジューラ.
            tracker: メトリクストラッカー.
            best_accuracy: 現在のベスト精度.
            get_learning_rate_fn: 現在の学習率を取得する関数.
            get_layer_wise_rates_fn: 層別学習率を取得する関数.
            is_layer_wise_lr_fn: 層別学習率が有効か判定する関数.

        Returns:
            tuple[float, bool]: (更新後のベスト精度, 訓練を停止すべきか).
        """
        train_metrics = train_epoch_fn(train_loader)

        val_metrics: Dict[str, float] = {}
        if val_loader:
            val_metrics = validate_fn(val_loader)

        if scheduler is not None:
            scheduler.step()

        # ログ出力
        self.logger.info(
            f"エポック {epoch} 完了 - "
            f"訓練損失: {train_metrics['loss']:.4f}, "
            f"訓練精度: {train_metrics['accuracy']:.2f}%"
        )

        should_stop = False
        if val_metrics:
            self.logger.info(
                f"検証損失: {val_metrics['val_loss']:.4f}, "
                f"検証精度: {val_metrics['val_accuracy']:.2f}%"
            )
            best_accuracy, should_stop = self._update_best_and_check_early_stop(
                epoch, val_metrics, model, optimizer, scheduler, best_accuracy
            )

        self._save_last_checkpoint(epoch, model, optimizer, scheduler, best_accuracy)
        self._record_metrics(
            tracker,
            epoch,
            train_metrics,
            val_metrics,
            model,
            get_learning_rate_fn,
            get_layer_wise_rates_fn,
            is_layer_wise_lr_fn,
        )

        return best_accuracy, should_stop

    def _update_best_and_check_early_stop(
        self,
        epoch: int,
        val_metrics: Dict[str, float],
        model: nn.Module,
        optimizer: Optional[optim.Optimizer],
        scheduler: Optional[optim.lr_scheduler.LRScheduler],
        best_accuracy: float,
    ) -> tuple[float, bool]:
        """ベストモデル更新とEarly Stopping判定.

        Args:
            epoch: 現在のエポック番号.
            val_metrics: 検証メトリクス.
            model: モデル.
            optimizer: オプティマイザ.
            scheduler: スケジューラ.
            best_accuracy: 現在のベスト精度.

        Returns:
            tuple[float, bool]: (更新後のベスト精度, 訓練を停止すべきか).
        """
        if val_metrics["val_accuracy"] >= best_accuracy:
            best_accuracy = val_metrics["val_accuracy"]
            self.checkpoint_store.save_best_model(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_accuracy=best_accuracy,
            )

        if self.early_stopping is not None:
            monitor = self.early_stopping.monitor
            monitor_value = val_metrics.get(monitor)
            if monitor_value is not None:
                if self.early_stopping.step(monitor_value, epoch):
                    return best_accuracy, True

        return best_accuracy, False

    def _save_last_checkpoint(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer],
        scheduler: Optional[optim.lr_scheduler.LRScheduler],
        best_accuracy: float,
    ) -> None:
        """現在の状態をラストチェックポイントとして保存."""
        self.checkpoint_store.save_last_model(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_accuracy=best_accuracy,
        )

    def _record_metrics(
        self,
        tracker: Optional[MetricsTracker],
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        model: nn.Module,
        get_learning_rate_fn: Callable[[], float],
        get_layer_wise_rates_fn: Callable[[], Dict[str, float]],
        is_layer_wise_lr_fn: Callable[[], bool],
    ) -> None:
        """メトリクスと勾配をトラッカーに記録.

        Args:
            tracker: メトリクストラッカー.
            epoch: 現在のエポック番号.
            train_metrics: 訓練メトリクス.
            val_metrics: 検証メトリクス.
            model: モデル.
            get_learning_rate_fn: 現在の学習率を取得する関数.
            get_layer_wise_rates_fn: 層別学習率を取得する関数.
            is_layer_wise_lr_fn: 層別学習率が有効か判定する関数.
        """
        if tracker is None:
            return

        layer_wise_rates: Dict[str, float] = {}
        if is_layer_wise_lr_fn():
            layer_rates = get_layer_wise_rates_fn()
            for layer_name, lr in layer_rates.items():
                layer_wise_rates[f"lr_{layer_name}"] = lr

        tracker.record_epoch(
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            model=model,
            learning_rate=get_learning_rate_fn(),
            layer_wise_lr_enabled=is_layer_wise_lr_fn(),
            layer_wise_rates=layer_wise_rates,
        )

    def _log_training_result(
        self,
        val_loader: Optional[DataLoader[Any]],
        last_epoch: int,
        best_accuracy: float,
    ) -> None:
        """訓練完了後のサマリーログを出力.

        Args:
            val_loader: 検証データローダー（ベスト精度表示の判定用）.
            last_epoch: 最終エポック番号.
            best_accuracy: ベスト精度.
        """
        if self.early_stopping is not None and self.early_stopping.should_stop:
            self.logger.info(
                f"Early Stoppingにより訓練を停止しました "
                f"(最終エポック: {last_epoch}, "
                f"ベスト{self.early_stopping.monitor}: "
                f"{self.early_stopping.best_value:.4f}, "
                f"ベストエポック: {self.early_stopping.best_epoch})"
            )
        else:
            self.logger.info("訓練が完了しました")
        if val_loader:
            self.logger.info(f"最高精度: {best_accuracy:.2f}%")

    def _finalize_metrics(self, tracker: Optional[MetricsTracker]) -> None:
        """訓練完了後のメトリクスエクスポートとサマリー表示.

        Args:
            tracker: メトリクストラッカー.
        """
        if tracker is None:
            return

        csv_path, graph_paths = tracker.finalize()
        if csv_path:
            self.logger.info(f"メトリクスCSVを出力: {csv_path}")
        if graph_paths:
            for graph_path in graph_paths:
                self.logger.info(f"メトリクスグラフを出力: {graph_path}")

        summary = tracker.get_summary()
        if summary:
            self.logger.info("=== 訓練サマリー ===")
            self.logger.info(f"総エポック数: {summary['total_epochs']}")
            self.logger.info(f"最終訓練損失: {summary['final_train_loss']:.4f}")
            self.logger.info(f"最終訓練精度: {summary['final_train_accuracy']:.2f}%")
            if "best_val_accuracy" in summary:
                self.logger.info(
                    f"最高検証精度: {summary['best_val_accuracy']:.2f}% "
                    f"(エポック {summary['best_val_accuracy_epoch']})"
                )

    @staticmethod
    def create_metrics_tracker(
        logger: logging.Logger,
        current_workspace: Optional[Path],
        visualization_dir: Optional[Path],
        enable_metrics_export: bool,
        enable_gradient_tracking: bool,
        gradient_tracking_config: Dict[str, Any],
        layer_wise_lr_graph_config: Dict[str, Any],
    ) -> Optional[MetricsTracker]:
        """MetricsTrackerを初期化.

        Args:
            logger: ロガーインスタンス.
            current_workspace: ワークスペースパス. Noneの場合トラッカーは作成しない.
            visualization_dir: 可視化出力ディレクトリ. 未作成時はNone.
            enable_metrics_export: メトリクスエクスポートを有効にするか.
            enable_gradient_tracking: 勾配追跡を有効にするか.
            gradient_tracking_config: 勾配追跡設定.
            layer_wise_lr_graph_config: 層別学習率グラフ設定.

        Returns:
            MetricsTracker: ワークスペースがある場合はトラッカー, なければNone.
        """
        if current_workspace is None or visualization_dir is None:
            return None

        tracker = MetricsTracker(
            logger=logger,
            visualization_dir=visualization_dir,
            enable_metrics_export=enable_metrics_export,
            enable_gradient_tracking=enable_gradient_tracking,
            gradient_tracking_config=gradient_tracking_config,
            layer_wise_lr_graph_config=layer_wise_lr_graph_config,
        )
        tracker.initialize()
        return tracker
