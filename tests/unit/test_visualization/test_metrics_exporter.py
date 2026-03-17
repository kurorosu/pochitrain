"""
TrainingMetricsExporterのテスト.
"""

from pathlib import Path

from pochitrain.visualization import TrainingMetricsExporter, metrics_exporter


class TestTrainingMetricsExporter:
    """TrainingMetricsExporterクラスのテスト."""

    def test_init(self, tmp_path):
        """初期化のテスト."""
        exporter = TrainingMetricsExporter(
            output_dir=tmp_path, enable_visualization=True
        )

        assert exporter.output_dir == tmp_path
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

    def test_record_epoch(self, tmp_path):
        """エポック記録のテスト."""
        exporter = TrainingMetricsExporter(output_dir=tmp_path)

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

    def test_record_multiple_epochs(self, tmp_path):
        """複数エポック記録のテスト."""
        exporter = TrainingMetricsExporter(output_dir=tmp_path)

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

    def test_export_all_csv(self, tmp_path):
        """export_all 経由の CSV 出力テスト."""
        exporter = TrainingMetricsExporter(
            output_dir=tmp_path, enable_visualization=False
        )

        for epoch in range(1, 4):
            exporter.record_epoch(
                epoch=epoch,
                learning_rate=0.001,
                train_loss=0.5,
                train_accuracy=85.0,
                val_loss=0.6,
                val_accuracy=83.0,
            )

        csv_path, graph_paths = exporter.export_all("test_metrics.csv")

        assert csv_path is not None
        assert csv_path.exists()
        assert csv_path.name == "test_metrics.csv"
        assert graph_paths is None

        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 4  # ヘッダー + 3エポック
            assert "epoch,learning_rate" in lines[0]

    def test_export_all_empty(self, tmp_path):
        """空の履歴での export_all テスト."""
        exporter = TrainingMetricsExporter(output_dir=tmp_path)

        csv_path, graph_paths = exporter.export_all()

        assert csv_path is None

    def test_export_all_with_graphs(self, tmp_path):
        """export_all 経由のグラフ生成テスト."""
        exporter = TrainingMetricsExporter(
            output_dir=tmp_path, enable_visualization=True
        )

        for epoch in range(1, 4):
            exporter.record_epoch(
                epoch=epoch,
                learning_rate=0.001,
                train_loss=1.0 / epoch,
                train_accuracy=80.0 + epoch,
                val_loss=1.2 / epoch,
                val_accuracy=78.0 + epoch,
            )

        csv_path, graph_paths = exporter.export_all()

        assert csv_path is not None
        assert graph_paths is not None
        assert len(graph_paths) == 2  # 損失、精度（学習率統合）の2つ
        assert all(p.exists() for p in graph_paths)
        assert any("loss" in str(p) for p in graph_paths)
        assert any("accuracy" in str(p) for p in graph_paths)

    def test_export_all_graphs_disabled(self, tmp_path):
        """グラフ生成が無効化された場合の export_all テスト."""
        exporter = TrainingMetricsExporter(
            output_dir=tmp_path, enable_visualization=False
        )

        exporter.record_epoch(
            epoch=1, learning_rate=0.001, train_loss=0.5, train_accuracy=85.0
        )

        csv_path, graph_paths = exporter.export_all()

        assert csv_path is not None
        assert graph_paths is None

    def test_export_all_with_layer_wise_lr(self, tmp_path, monkeypatch):
        """層別学習率有効時に専用グラフが追加生成されることを確認."""

        def _fast_savefig(path, *args, **kwargs):
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"test")

        exporter = TrainingMetricsExporter(
            output_dir=tmp_path, enable_visualization=True
        )

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

        monkeypatch.setattr(metrics_exporter.plt, "savefig", _fast_savefig)
        csv_path, graph_paths = exporter.export_all()

        assert graph_paths is not None
        assert len(graph_paths) == 3  # 損失、精度、層別学習率の3つ
        assert all(p.exists() for p in graph_paths)
        assert any("layer_wise_lr" in str(p) for p in graph_paths)

    def test_get_summary_includes_best_epoch(self, tmp_path):
        """get_summary が最良エポック情報を含むことを検証."""
        exporter = TrainingMetricsExporter(output_dir=tmp_path)

        exporter.record_epoch(
            epoch=1,
            learning_rate=0.001,
            train_loss=0.5,
            train_accuracy=85.0,
            val_loss=0.6,
            val_accuracy=80.0,
        )
        exporter.record_epoch(
            epoch=2,
            learning_rate=0.001,
            train_loss=0.4,
            train_accuracy=87.0,
            val_loss=0.5,
            val_accuracy=85.0,
        )
        exporter.record_epoch(
            epoch=3,
            learning_rate=0.001,
            train_loss=0.3,
            train_accuracy=90.0,
            val_loss=0.4,
            val_accuracy=88.0,
        )
        exporter.record_epoch(
            epoch=4,
            learning_rate=0.001,
            train_loss=0.35,
            train_accuracy=89.0,
            val_loss=0.45,
            val_accuracy=86.0,
        )

        summary = exporter.get_summary()

        assert summary["best_val_accuracy"] == 88.0
        assert summary["best_val_accuracy_epoch"] == 3

    def test_get_summary(self, tmp_path):
        """サマリー取得のテスト."""
        exporter = TrainingMetricsExporter(output_dir=tmp_path)

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

    def test_record_with_layer_wise_lr_adds_headers(self, tmp_path):
        """層別学習率記録時に拡張ヘッダーが自動追加される."""
        exporter = TrainingMetricsExporter(output_dir=tmp_path)

        exporter.record_epoch(
            epoch=1,
            learning_rate=0.001,
            train_loss=0.5,
            train_accuracy=85.0,
            layer_wise_lr_enabled=True,
            lr_backbone=0.0001,
            lr_head=0.001,
        )

        assert "lr_backbone" in exporter.extended_headers
        assert "lr_head" in exporter.extended_headers
        assert exporter.metrics_history[0]["lr_backbone"] == 0.0001
        assert exporter.metrics_history[0]["lr_head"] == 0.001

    def test_record_without_validation_data(self, tmp_path):
        """検証データなしでの記録・エクスポートテスト."""
        exporter = TrainingMetricsExporter(output_dir=tmp_path)

        exporter.record_epoch(
            epoch=1, learning_rate=0.001, train_loss=0.5, train_accuracy=85.0
        )

        assert exporter.metrics_history[0]["val_loss"] == ""
        assert exporter.metrics_history[0]["val_accuracy"] == ""

        csv_path, _ = exporter.export_all()
        assert csv_path is not None
        assert csv_path.exists()
