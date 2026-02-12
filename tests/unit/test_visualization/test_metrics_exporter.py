"""
TrainingMetricsExporterのテスト.
"""

import tempfile
from pathlib import Path

from pochitrain.visualization import TrainingMetricsExporter


class TestTrainingMetricsExporter:
    """TrainingMetricsExporterクラスのテスト."""

    def test_init(self):
        """初期化のテスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = TrainingMetricsExporter(
                output_dir=Path(temp_dir), enable_visualization=True
            )

            assert exporter.output_dir == Path(temp_dir)
            assert exporter.enable_visualization is True
            assert len(exporter.metrics_history) == 0
            assert exporter.base_headers == [
                "epoch",
                "learning_rate",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
            ]

    def test_record_epoch(self):
        """エポック記録のテスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = TrainingMetricsExporter(output_dir=Path(temp_dir))

            exporter.record_epoch(
                epoch=1,
                learning_rate=0.001,
                train_loss=0.5,
                train_accuracy=85.0,
                val_loss=0.6,
                val_accuracy=83.0,
            )

            assert len(exporter.metrics_history) == 1
            assert exporter.metrics_history[0]["epoch"] == 1
            assert exporter.metrics_history[0]["learning_rate"] == 0.001
            assert exporter.metrics_history[0]["train_loss"] == 0.5
            assert exporter.metrics_history[0]["train_accuracy"] == 85.0
            assert exporter.metrics_history[0]["val_loss"] == 0.6
            assert exporter.metrics_history[0]["val_accuracy"] == 83.0

    def test_record_multiple_epochs(self):
        """複数エポック記録のテスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = TrainingMetricsExporter(output_dir=Path(temp_dir))

            for epoch in range(1, 6):
                exporter.record_epoch(
                    epoch=epoch,
                    learning_rate=0.001 / epoch,
                    train_loss=1.0 / epoch,
                    train_accuracy=80.0 + epoch,
                )

            assert len(exporter.metrics_history) == 5
            assert exporter.metrics_history[0]["epoch"] == 1
            assert exporter.metrics_history[4]["epoch"] == 5
            assert exporter.metrics_history[4]["train_accuracy"] == 85.0

    def test_export_to_csv(self):
        """CSV出力のテスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = TrainingMetricsExporter(output_dir=Path(temp_dir))

            # メトリクスを記録
            for epoch in range(1, 4):
                exporter.record_epoch(
                    epoch=epoch,
                    learning_rate=0.001,
                    train_loss=0.5,
                    train_accuracy=85.0,
                    val_loss=0.6,
                    val_accuracy=83.0,
                )

            # CSVに出力
            csv_path = exporter.export_to_csv("test_metrics.csv")

            assert csv_path is not None
            assert csv_path.exists()
            assert csv_path.name == "test_metrics.csv"

            # CSVファイルの内容確認
            with open(csv_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                assert len(lines) == 4  # ヘッダー + 3エポック
                assert "epoch,learning_rate" in lines[0]

    def test_export_to_csv_empty(self):
        """空の履歴でのCSV出力テスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = TrainingMetricsExporter(output_dir=Path(temp_dir))

            csv_path = exporter.export_to_csv()

            assert csv_path is None

    def test_generate_graphs(self):
        """グラフ生成のテスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = TrainingMetricsExporter(
                output_dir=Path(temp_dir), enable_visualization=True
            )

            # メトリクスを記録
            for epoch in range(1, 6):
                exporter.record_epoch(
                    epoch=epoch,
                    learning_rate=0.001,
                    train_loss=1.0 / epoch,
                    train_accuracy=80.0 + epoch,
                    val_loss=1.2 / epoch,
                    val_accuracy=78.0 + epoch,
                )

            # グラフを生成
            graph_paths = exporter.generate_graphs("test_graph")

            assert graph_paths is not None
            assert len(graph_paths) == 2  # 損失、精度（学習率統合）の2つ
            assert all(p.exists() for p in graph_paths)
            assert any("loss" in str(p) for p in graph_paths)
            assert any("accuracy" in str(p) for p in graph_paths)

    def test_generate_graphs_disabled(self):
        """グラフ生成が無効化された場合のテスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = TrainingMetricsExporter(
                output_dir=Path(temp_dir), enable_visualization=False
            )

            exporter.record_epoch(
                epoch=1, learning_rate=0.001, train_loss=0.5, train_accuracy=85.0
            )

            graph_paths = exporter.generate_graphs()

            assert graph_paths is None

    def test_generate_graphs_with_layer_wise_lr(self):
        """層別学習率有効時に専用グラフが追加生成されることを確認."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = TrainingMetricsExporter(
                output_dir=Path(temp_dir), enable_visualization=True
            )

            # 層別学習率付きメトリクスを記録
            for epoch in range(1, 4):
                exporter.record_epoch(
                    epoch=epoch,
                    learning_rate=0.001,
                    train_loss=0.5,
                    train_accuracy=85.0,
                    layer_wise_lr_enabled=True,
                    lr_backbone=0.0001 * epoch,
                    lr_head=0.001 * epoch,
                )

            graph_paths = exporter.generate_graphs("test_layer_wise")
            assert graph_paths is not None
            assert len(graph_paths) == 3  # 損失、精度、層別学習率の3つ
            assert all(p.exists() for p in graph_paths)
            assert any("layer_wise_lr" in str(p) for p in graph_paths)

    def test_get_best_epoch(self):
        """最良エポック取得のテスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = TrainingMetricsExporter(output_dir=Path(temp_dir))

            # メトリクスを記録（エポック3が最高精度）
            exporter.record_epoch(
                epoch=1,
                learning_rate=0.001,
                train_loss=0.5,
                train_accuracy=85.0,
                val_accuracy=80.0,
            )
            exporter.record_epoch(
                epoch=2,
                learning_rate=0.001,
                train_loss=0.4,
                train_accuracy=87.0,
                val_accuracy=85.0,
            )
            exporter.record_epoch(
                epoch=3,
                learning_rate=0.001,
                train_loss=0.3,
                train_accuracy=90.0,
                val_accuracy=88.0,
            )
            exporter.record_epoch(
                epoch=4,
                learning_rate=0.001,
                train_loss=0.35,
                train_accuracy=89.0,
                val_accuracy=86.0,
            )

            best = exporter.get_best_epoch("val_accuracy")

            assert best is not None
            assert best["epoch"] == 3
            assert best["val_accuracy"] == 88.0

    def test_get_summary(self):
        """サマリー取得のテスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = TrainingMetricsExporter(output_dir=Path(temp_dir))

            # メトリクスを記録
            for epoch in range(1, 6):
                exporter.record_epoch(
                    epoch=epoch,
                    learning_rate=0.001,
                    train_loss=1.0 / epoch,
                    train_accuracy=80.0 + epoch,
                    val_loss=1.2 / epoch,
                    val_accuracy=78.0 + epoch,
                )

            summary = exporter.get_summary()

            assert summary["total_epochs"] == 5
            assert summary["final_train_loss"] == 1.0 / 5
            assert summary["final_train_accuracy"] == 85.0
            assert summary["final_val_loss"] == 1.2 / 5
            assert summary["final_val_accuracy"] == 83.0
            assert summary["best_val_accuracy"] == 83.0
            assert summary["best_val_accuracy_epoch"] == 5

    def test_add_extended_headers(self):
        """拡張ヘッダー追加のテスト（Issue 9用）."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = TrainingMetricsExporter(output_dir=Path(temp_dir))

            exporter.add_extended_headers(["param_1", "param_2"])

            assert "param_1" in exporter.extended_headers
            assert "param_2" in exporter.extended_headers

            # 拡張メトリクス付きで記録
            exporter.record_epoch(
                epoch=1,
                learning_rate=0.001,
                train_loss=0.5,
                train_accuracy=85.0,
                param_1=0.123,
                param_2=0.456,
            )

            assert exporter.metrics_history[0]["param_1"] == 0.123
            assert exporter.metrics_history[0]["param_2"] == 0.456

    def test_record_without_validation_data(self):
        """検証データなしでの記録テスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = TrainingMetricsExporter(output_dir=Path(temp_dir))

            exporter.record_epoch(
                epoch=1, learning_rate=0.001, train_loss=0.5, train_accuracy=85.0
            )

            assert exporter.metrics_history[0]["val_loss"] == ""
            assert exporter.metrics_history[0]["val_accuracy"] == ""

            # CSVに出力して確認
            csv_path = exporter.export_to_csv()
            assert csv_path is not None
            assert csv_path.exists()
