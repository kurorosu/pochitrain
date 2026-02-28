"""MetricsTrackerクラスのユニットテスト."""

import logging
from pathlib import Path

import pytest
import torch
from torch import nn

from pochitrain.training.metrics_tracker import MetricsTracker


@pytest.fixture
def visualization_dir(tmp_path: Path) -> Path:
    """テスト用の可視化出力ディレクトリ."""
    return tmp_path


@pytest.fixture
def model() -> nn.Module:
    """テスト用の簡易モデル."""
    m = nn.Linear(10, 2)
    x = torch.randn(1, 10)
    y = m(x)
    y.sum().backward()
    return m


class TestInitialize:
    """initializeメソッドのテスト."""

    def test_initialize_metrics_exporter(
        self, logger: logging.Logger, visualization_dir: Path
    ) -> None:
        """enable_metrics_export=Trueの場合, MetricsExporterが初期化される."""
        tracker = MetricsTracker(
            logger=logger,
            visualization_dir=visualization_dir,
            enable_metrics_export=True,
            enable_gradient_tracking=False,
        )
        tracker.initialize()

        assert tracker._metrics_exporter is not None
        assert tracker._gradient_tracer is None

    def test_initialize_gradient_tracer(
        self, logger: logging.Logger, visualization_dir: Path
    ) -> None:
        """enable_gradient_tracking=Trueの場合, GradientTracerが初期化される."""
        tracker = MetricsTracker(
            logger=logger,
            visualization_dir=visualization_dir,
            enable_metrics_export=False,
            enable_gradient_tracking=True,
        )
        tracker.initialize()

        assert tracker._metrics_exporter is None
        assert tracker._gradient_tracer is not None

    def test_initialize_both(
        self, logger: logging.Logger, visualization_dir: Path
    ) -> None:
        """両方有効の場合, 両方初期化される."""
        tracker = MetricsTracker(
            logger=logger,
            visualization_dir=visualization_dir,
            enable_metrics_export=True,
            enable_gradient_tracking=True,
        )
        tracker.initialize()

        assert tracker._metrics_exporter is not None
        assert tracker._gradient_tracer is not None

    def test_initialize_disabled(
        self, logger: logging.Logger, visualization_dir: Path
    ) -> None:
        """両方無効の場合, 何も初期化されない."""
        tracker = MetricsTracker(
            logger=logger,
            visualization_dir=visualization_dir,
            enable_metrics_export=False,
            enable_gradient_tracking=False,
        )
        tracker.initialize()

        assert tracker._metrics_exporter is None
        assert tracker._gradient_tracer is None

    def test_initialize_gradient_tracer_config(
        self, logger: logging.Logger, visualization_dir: Path
    ) -> None:
        """勾配トラッキング設定が正しく反映される."""
        config = {
            "exclude_patterns": [r"layer1\."],
            "group_by_block": False,
            "aggregation_method": "mean",
        }
        tracker = MetricsTracker(
            logger=logger,
            visualization_dir=visualization_dir,
            enable_gradient_tracking=True,
            gradient_tracking_config=config,
        )
        tracker.initialize()

        assert tracker._gradient_tracer is not None
        assert tracker._gradient_tracer.group_by_block is False
        assert tracker._gradient_tracer.aggregation_method == "mean"


class TestRecordEpoch:
    """record_epochメソッドのテスト."""

    def test_record_epoch_metrics(
        self, logger: logging.Logger, visualization_dir: Path, model: nn.Module
    ) -> None:
        """メトリクスが記録される."""
        tracker = MetricsTracker(
            logger=logger,
            visualization_dir=visualization_dir,
            enable_metrics_export=True,
            enable_gradient_tracking=False,
        )
        tracker.initialize()

        tracker.record_epoch(
            epoch=1,
            train_metrics={"loss": 0.5, "accuracy": 80.0},
            val_metrics={"val_loss": 0.6, "val_accuracy": 75.0},
            model=model,
            learning_rate=0.001,
        )

        assert tracker._metrics_exporter is not None
        assert len(tracker._metrics_exporter.metrics_history) == 1

    def test_record_epoch_gradient(
        self, logger: logging.Logger, visualization_dir: Path, model: nn.Module
    ) -> None:
        """勾配が記録される."""
        tracker = MetricsTracker(
            logger=logger,
            visualization_dir=visualization_dir,
            enable_metrics_export=False,
            enable_gradient_tracking=True,
        )
        tracker.initialize()

        tracker.record_epoch(
            epoch=1,
            train_metrics={"loss": 0.5, "accuracy": 80.0},
            val_metrics={},
            model=model,
            learning_rate=0.001,
        )

        assert tracker._gradient_tracer is not None
        assert len(tracker._gradient_tracer.epochs) == 1

    def test_record_epoch_gradient_frequency(
        self, logger: logging.Logger, visualization_dir: Path, model: nn.Module
    ) -> None:
        """記録頻度の制御が正しく動作する."""
        tracker = MetricsTracker(
            logger=logger,
            visualization_dir=visualization_dir,
            enable_metrics_export=False,
            enable_gradient_tracking=True,
            gradient_tracking_config={"record_frequency": 2},
        )
        tracker.initialize()

        tracker.record_epoch(
            epoch=1,
            train_metrics={"loss": 0.5, "accuracy": 80.0},
            val_metrics={},
            model=model,
            learning_rate=0.001,
        )
        assert len(tracker._gradient_tracer.epochs) == 0  # type: ignore[union-attr]

        tracker.record_epoch(
            epoch=2,
            train_metrics={"loss": 0.4, "accuracy": 85.0},
            val_metrics={},
            model=model,
            learning_rate=0.001,
        )
        assert len(tracker._gradient_tracer.epochs) == 1  # type: ignore[union-attr]

    def test_record_epoch_with_layer_wise_lr(
        self, logger: logging.Logger, visualization_dir: Path, model: nn.Module
    ) -> None:
        """層別学習率情報付きでメトリクスが記録される."""
        tracker = MetricsTracker(
            logger=logger,
            visualization_dir=visualization_dir,
            enable_metrics_export=True,
            enable_gradient_tracking=False,
        )
        tracker.initialize()

        tracker.record_epoch(
            epoch=1,
            train_metrics={"loss": 0.5, "accuracy": 80.0},
            val_metrics={"val_loss": 0.6, "val_accuracy": 75.0},
            model=model,
            learning_rate=0.001,
            layer_wise_lr_enabled=True,
            layer_wise_rates={"lr_layer1": 0.0001, "lr_fc": 0.001},
        )

        assert tracker._metrics_exporter is not None
        recorded = tracker._metrics_exporter.metrics_history[0]
        assert recorded["lr_layer1"] == 0.0001
        assert recorded["lr_fc"] == 0.001

    def test_record_epoch_disabled(
        self, logger: logging.Logger, visualization_dir: Path, model: nn.Module
    ) -> None:
        """両方無効の場合, エラーなく処理される."""
        tracker = MetricsTracker(
            logger=logger,
            visualization_dir=visualization_dir,
            enable_metrics_export=False,
            enable_gradient_tracking=False,
        )
        tracker.initialize()

        tracker.record_epoch(
            epoch=1,
            train_metrics={"loss": 0.5, "accuracy": 80.0},
            val_metrics={},
            model=model,
            learning_rate=0.001,
        )
        assert tracker._metrics_exporter is None
        assert tracker._gradient_tracer is None
        assert list(visualization_dir.glob("gradient_trace_*.csv")) == []


class TestFinalize:
    """finalizeメソッドのテスト."""

    def test_finalize_with_metrics(
        self, logger: logging.Logger, visualization_dir: Path, model: nn.Module
    ) -> None:
        """CSVとグラフが出力される."""
        tracker = MetricsTracker(
            logger=logger,
            visualization_dir=visualization_dir,
            enable_metrics_export=True,
            enable_gradient_tracking=False,
        )
        tracker.initialize()

        tracker.record_epoch(
            epoch=1,
            train_metrics={"loss": 0.5, "accuracy": 80.0},
            val_metrics={"val_loss": 0.6, "val_accuracy": 75.0},
            model=model,
            learning_rate=0.001,
        )

        csv_path, graph_paths = tracker.finalize()

        assert csv_path is not None
        assert csv_path.exists()
        assert csv_path.suffix == ".csv"

    def test_finalize_disabled(
        self, logger: logging.Logger, visualization_dir: Path
    ) -> None:
        """両方無効の場合, Noneと空リストが返される."""
        tracker = MetricsTracker(
            logger=logger,
            visualization_dir=visualization_dir,
            enable_metrics_export=False,
            enable_gradient_tracking=False,
        )
        tracker.initialize()

        csv_path, graph_paths = tracker.finalize()
        assert csv_path is None
        assert graph_paths == []

    def test_finalize_gradient_csv(
        self, logger: logging.Logger, visualization_dir: Path, model: nn.Module
    ) -> None:
        """勾配CSVが出力される."""
        tracker = MetricsTracker(
            logger=logger,
            visualization_dir=visualization_dir,
            enable_metrics_export=False,
            enable_gradient_tracking=True,
        )
        tracker.initialize()

        tracker.record_epoch(
            epoch=1,
            train_metrics={"loss": 0.5, "accuracy": 80.0},
            val_metrics={},
            model=model,
            learning_rate=0.001,
        )

        tracker.finalize()

        gradient_csvs = list(visualization_dir.glob("gradient_trace_*.csv"))
        assert len(gradient_csvs) == 1


class TestGetSummary:
    """get_summaryメソッドのテスト."""

    def test_get_summary(
        self, logger: logging.Logger, visualization_dir: Path, model: nn.Module
    ) -> None:
        """サマリーが取得できる."""
        tracker = MetricsTracker(
            logger=logger,
            visualization_dir=visualization_dir,
            enable_metrics_export=True,
            enable_gradient_tracking=False,
        )
        tracker.initialize()

        tracker.record_epoch(
            epoch=1,
            train_metrics={"loss": 0.5, "accuracy": 80.0},
            val_metrics={"val_loss": 0.6, "val_accuracy": 75.0},
            model=model,
            learning_rate=0.001,
        )

        summary = tracker.get_summary()
        assert summary is not None
        assert "total_epochs" in summary
        assert "final_train_loss" in summary
        assert "final_train_accuracy" in summary

    def test_get_summary_disabled(
        self, logger: logging.Logger, visualization_dir: Path
    ) -> None:
        """メトリクスエクスポーター無効の場合, Noneが返される."""
        tracker = MetricsTracker(
            logger=logger,
            visualization_dir=visualization_dir,
            enable_metrics_export=False,
        )
        tracker.initialize()

        summary = tracker.get_summary()
        assert summary is None

    def test_get_summary_no_records(
        self, logger: logging.Logger, visualization_dir: Path
    ) -> None:
        """記録がない場合, Noneが返される."""
        tracker = MetricsTracker(
            logger=logger,
            visualization_dir=visualization_dir,
            enable_metrics_export=True,
        )
        tracker.initialize()

        summary = tracker.get_summary()
        assert summary is None
