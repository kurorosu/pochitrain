"""訓練メトリクスと勾配の記録・エクスポートを統括するモジュール."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from torch import nn

from pochitrain.utils.timestamp_utils import get_current_timestamp
from pochitrain.visualization import GradientTracer, TrainingMetricsExporter


class MetricsTracker:
    """訓練メトリクスと勾配の記録・エクスポートを統括するクラス.

    Facade パターンに基づき, TrainingMetricsExporter と GradientTracer の
    統一インターフェースを提供する.

    Args:
        logger: ロガーインスタンス
        visualization_dir: 可視化出力ディレクトリ
        enable_metrics_export: メトリクスエクスポートを有効にするか
        enable_gradient_tracking: 勾配トラッキングを有効にするか
        gradient_tracking_config: 勾配トラッキングの設定
        layer_wise_lr_graph_config: 層別学習率グラフの設定
    """

    def __init__(
        self,
        logger: logging.Logger,
        visualization_dir: Path,
        enable_metrics_export: bool = True,
        enable_gradient_tracking: bool = False,
        gradient_tracking_config: Optional[Dict[str, Any]] = None,
        layer_wise_lr_graph_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """MetricsTrackerを初期化."""
        self.logger = logger
        self.visualization_dir = visualization_dir
        self.enable_metrics_export = enable_metrics_export
        self.enable_gradient_tracking = enable_gradient_tracking
        self.gradient_tracking_config = gradient_tracking_config or {}
        self.layer_wise_lr_graph_config = layer_wise_lr_graph_config

        # 条件的に初期化されるコンポーネント
        self._metrics_exporter: Optional[TrainingMetricsExporter] = None
        self._gradient_tracer: Optional[GradientTracer] = None

    def initialize(self) -> None:
        """メトリクスエクスポーターと勾配トレーサーを初期化."""
        if self.enable_metrics_export:
            self._metrics_exporter = TrainingMetricsExporter(
                output_dir=self.visualization_dir,
                enable_visualization=True,
                logger=self.logger,
                layer_wise_lr_graph_config=self.layer_wise_lr_graph_config,
            )
            self.logger.debug("メトリクス記録機能を有効化しました")

        if self.enable_gradient_tracking:
            exclude_patterns = self.gradient_tracking_config.get(
                "exclude_patterns", ["fc\\.", "\\.bias"]
            )
            group_by_block = self.gradient_tracking_config.get("group_by_block", True)
            aggregation_method = self.gradient_tracking_config.get(
                "aggregation_method", "median"
            )

            self._gradient_tracer = GradientTracer(
                logger=self.logger,
                exclude_patterns=exclude_patterns,
                group_by_block=group_by_block,
                aggregation_method=aggregation_method,
            )
            self.logger.debug(
                f"勾配トレース機能を有効化しました "
                f"(集約: {aggregation_method}, ブロック化: {group_by_block})"
            )

    def record_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        model: nn.Module,
        learning_rate: float,
        layer_wise_lr_enabled: bool = False,
        layer_wise_rates: Optional[Dict[str, float]] = None,
    ) -> None:
        """エポックのメトリクスと勾配を記録.

        Args:
            epoch: エポック番号
            train_metrics: 訓練メトリクス (loss, accuracy)
            val_metrics: 検証メトリクス (val_loss, val_accuracy)
            model: モデル (勾配記録用)
            learning_rate: 現在の学習率
            layer_wise_lr_enabled: 層別学習率が有効か
            layer_wise_rates: 層別学習率の辞書 (例: {"lr_layer1": 0.001})
        """
        # メトリクスの記録
        if self._metrics_exporter is not None:
            kwargs: Dict[str, float] = {}
            if layer_wise_rates is not None:
                kwargs = layer_wise_rates

            self._metrics_exporter.record_epoch(
                epoch=epoch,
                learning_rate=learning_rate,
                train_loss=train_metrics["loss"],
                train_accuracy=train_metrics["accuracy"],
                val_loss=val_metrics.get("val_loss"),
                val_accuracy=val_metrics.get("val_accuracy"),
                layer_wise_lr_enabled=layer_wise_lr_enabled,
                **kwargs,
            )

        # 勾配ノルムの記録
        if self._gradient_tracer is not None:
            record_freq = self.gradient_tracking_config.get("record_frequency", 1)
            if epoch % record_freq == 0:
                self._gradient_tracer.record_gradients(model, epoch)

    def finalize(self) -> Tuple[Optional[Path], List[Path]]:
        """訓練完了後のエクスポート処理. CSVとグラフを出力.

        Returns:
            (csv_path, graph_paths) のタプル.
            csv_path: メトリクスCSVのパス (無効時は None)
            graph_paths: グラフファイルパスのリスト (無効時は空リスト)
        """
        csv_path: Optional[Path] = None
        graph_paths: List[Path] = []

        # メトリクスのエクスポート
        if self._metrics_exporter is not None:
            exported_csv, exported_graphs = self._metrics_exporter.export_all()
            csv_path = exported_csv
            if exported_graphs:
                graph_paths = exported_graphs

        # 勾配トレースをCSVに保存
        if self._gradient_tracer is not None:
            timestamp = get_current_timestamp()
            gradient_csv_path = (
                self.visualization_dir / f"gradient_trace_{timestamp}.csv"
            )
            self._gradient_tracer.save_csv(gradient_csv_path)

        return csv_path, graph_paths

    def get_summary(self) -> Optional[Dict[str, Any]]:
        """訓練サマリーを取得.

        Returns:
            サマリー情報の辞書. メトリクスエクスポーターが無効の場合は None.
        """
        if self._metrics_exporter is None:
            return None

        summary = self._metrics_exporter.get_summary()
        return summary if summary else None
