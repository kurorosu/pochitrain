"""InferenceResultExporterクラスのテスト."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pochitrain.inference.result_exporter import InferenceResultExporter
from pochitrain.utils.directory_manager import InferenceWorkspaceManager


class TestInferenceResultExporterInit:
    """InferenceResultExporter 初期化のテスト."""

    def test_basic_init(self, tmp_path):
        """基本的な初期化が成功する."""
        manager = InferenceWorkspaceManager(str(tmp_path / "inference_results"))
        logger = MagicMock()
        exporter = InferenceResultExporter(workspace_manager=manager, logger=logger)
        assert exporter.workspace is None
        assert exporter.workspace_manager is manager

    def test_workspace_not_created_on_init(self, tmp_path):
        """初期化時にワークスペースが作成されない(遅延作成)."""
        manager = InferenceWorkspaceManager(str(tmp_path / "inference_results"))
        logger = MagicMock()
        exporter = InferenceResultExporter(workspace_manager=manager, logger=logger)
        assert exporter.workspace is None


class TestInferenceResultExporterEnsureWorkspace:
    """_ensure_workspace メソッドのテスト."""

    def test_creates_workspace_on_first_call(self, tmp_path):
        """初回呼び出しでワークスペースが作成される."""
        manager = InferenceWorkspaceManager(str(tmp_path / "inference_results"))
        logger = MagicMock()
        exporter = InferenceResultExporter(workspace_manager=manager, logger=logger)
        assert exporter.workspace is None

        workspace = exporter._ensure_workspace()
        assert workspace is not None
        assert workspace.exists()
        assert exporter.workspace == workspace

    def test_returns_same_workspace_on_subsequent_calls(self, tmp_path):
        """2回目以降は同じワークスペースを返す."""
        manager = InferenceWorkspaceManager(str(tmp_path / "inference_results"))
        logger = MagicMock()
        exporter = InferenceResultExporter(workspace_manager=manager, logger=logger)
        workspace1 = exporter._ensure_workspace()
        workspace2 = exporter._ensure_workspace()
        assert workspace1 == workspace2


class TestInferenceResultExporterGetWorkspaceInfo:
    """get_workspace_info メソッドのテスト."""

    def test_no_workspace_created(self, tmp_path):
        """ワークスペース未作成時の情報."""
        manager = InferenceWorkspaceManager(str(tmp_path / "inference_results"))
        logger = MagicMock()
        exporter = InferenceResultExporter(workspace_manager=manager, logger=logger)
        info = exporter.get_workspace_info()
        assert info["workspace"] is None
        assert info["exists"] is False

    def test_workspace_created(self, tmp_path):
        """ワークスペース作成後の情報."""
        manager = InferenceWorkspaceManager(str(tmp_path / "inference_results"))
        logger = MagicMock()
        exporter = InferenceResultExporter(workspace_manager=manager, logger=logger)
        exporter._ensure_workspace()
        info = exporter.get_workspace_info()
        assert info["workspace"] is not None
        assert info["exists"] is True


class TestInferenceResultExporterExport:
    """export メソッドのテスト."""

    @pytest.fixture()
    def sample_data(self):
        """テスト用サンプルデータ."""
        return {
            "image_paths": ["/img/a.jpg", "/img/b.jpg", "/img/c.jpg"],
            "predicted_labels": [0, 1, 0],
            "true_labels": [0, 1, 1],
            "confidence_scores": [0.95, 0.80, 0.60],
            "class_names": ["cat", "dog"],
            "model_info": {
                "model_name": "resnet18",
                "num_classes": 2,
                "device": "cpu",
                "model_path": "/path/to/model.pth",
                "best_accuracy": 90.0,
                "epoch": 10,
            },
        }

    def test_export_creates_csv_files(self, tmp_path, sample_data):
        """CSV出力が行われる."""
        manager = InferenceWorkspaceManager(str(tmp_path / "inference_results"))
        logger = MagicMock()
        exporter = InferenceResultExporter(workspace_manager=manager, logger=logger)

        with (
            patch.object(
                exporter, "save_confusion_matrix_image", return_value=Path("cm.png")
            ),
            patch(
                "pochitrain.utils.inference_utils.save_classification_report",
                return_value=Path("report.csv"),
            ),
        ):
            results_csv, summary_csv = exporter.export(**sample_data)

        assert results_csv.exists()
        assert summary_csv.exists()
        assert results_csv.name == "inference_results.csv"
        assert summary_csv.name == "inference_summary.csv"

    def test_export_creates_workspace_lazily(self, tmp_path, sample_data):
        """export 時にワークスペースが遅延作成される."""
        manager = InferenceWorkspaceManager(str(tmp_path / "inference_results"))
        logger = MagicMock()
        exporter = InferenceResultExporter(workspace_manager=manager, logger=logger)
        assert exporter.workspace is None

        with (
            patch.object(
                exporter, "save_confusion_matrix_image", return_value=Path("cm.png")
            ),
            patch(
                "pochitrain.utils.inference_utils.save_classification_report",
                return_value=Path("report.csv"),
            ),
        ):
            exporter.export(**sample_data)

        assert exporter.workspace is not None
        assert exporter.workspace.exists()

    def test_export_saves_model_info_json(self, tmp_path, sample_data):
        """モデル情報JSONが保存される."""
        manager = InferenceWorkspaceManager(str(tmp_path / "inference_results"))
        logger = MagicMock()
        exporter = InferenceResultExporter(workspace_manager=manager, logger=logger)

        with (
            patch.object(
                exporter, "save_confusion_matrix_image", return_value=Path("cm.png")
            ),
            patch(
                "pochitrain.utils.inference_utils.save_classification_report",
                return_value=Path("report.csv"),
            ),
        ):
            exporter.export(**sample_data)

        # model_info.json が保存されていることを確認
        workspace = exporter.workspace
        assert workspace is not None
        model_info_path = workspace / "model_info.json"
        assert model_info_path.exists()

    def test_export_handles_confusion_matrix_failure_gracefully(
        self, tmp_path, sample_data
    ):
        """混同行列画像生成失敗時もエクスポートが完了する."""
        manager = InferenceWorkspaceManager(str(tmp_path / "inference_results"))
        logger = MagicMock()
        exporter = InferenceResultExporter(workspace_manager=manager, logger=logger)

        with (
            patch.object(
                exporter,
                "save_confusion_matrix_image",
                side_effect=RuntimeError("matplotlib error"),
            ),
            patch(
                "pochitrain.utils.inference_utils.save_classification_report",
                return_value=Path("report.csv"),
            ),
        ):
            results_csv, summary_csv = exporter.export(**sample_data)

        # エラーがあってもCSV出力は成功
        assert results_csv.exists()
        assert summary_csv.exists()
        # 警告ログが出力されている
        logger.warning.assert_any_call(
            "混同行列画像生成に失敗しました: matplotlib error"
        )

    def test_export_handles_classification_report_failure_gracefully(
        self, tmp_path, sample_data
    ):
        """クラス別精度レポート生成失敗時もエクスポートが完了する."""
        manager = InferenceWorkspaceManager(str(tmp_path / "inference_results"))
        logger = MagicMock()
        exporter = InferenceResultExporter(workspace_manager=manager, logger=logger)

        with (
            patch.object(
                exporter, "save_confusion_matrix_image", return_value=Path("cm.png")
            ),
            patch(
                "pochitrain.utils.inference_utils.save_classification_report",
                side_effect=RuntimeError("sklearn error"),
            ),
        ):
            results_csv, summary_csv = exporter.export(**sample_data)

        assert results_csv.exists()
        assert summary_csv.exists()
        logger.warning.assert_any_call(
            "クラス別精度レポート生成に失敗しました: sklearn error"
        )


class TestInferenceResultExporterSaveConfusionMatrixImage:
    """save_confusion_matrix_image メソッドのテスト."""

    def test_saves_image(self, tmp_path):
        """混同行列画像が保存される."""
        manager = InferenceWorkspaceManager(str(tmp_path / "inference_results"))
        logger = MagicMock()
        exporter = InferenceResultExporter(workspace_manager=manager, logger=logger)

        result_path = exporter.save_confusion_matrix_image(
            predicted_labels=[0, 1, 0],
            true_labels=[0, 1, 1],
            class_names=["cat", "dog"],
        )

        assert result_path.exists()
        assert result_path.name == "confusion_matrix.png"

    def test_saves_image_with_custom_filename(self, tmp_path):
        """カスタムファイル名で混同行列画像が保存される."""
        manager = InferenceWorkspaceManager(str(tmp_path / "inference_results"))
        logger = MagicMock()
        exporter = InferenceResultExporter(workspace_manager=manager, logger=logger)

        result_path = exporter.save_confusion_matrix_image(
            predicted_labels=[0, 1, 0],
            true_labels=[0, 1, 1],
            class_names=["cat", "dog"],
            filename="custom_cm.png",
        )

        assert result_path.exists()
        assert result_path.name == "custom_cm.png"
