"""ResultExporterのユニットテスト."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from pochitrain.optimization.result_exporter import (
    ConfigExporter,
    JsonResultExporter,
)


class TestJsonResultExporter:
    """JsonResultExporterのテスト."""

    def test_export_creates_best_params_json(self) -> None:
        """best_params.jsonが作成されることをテスト."""
        exporter = JsonResultExporter()

        # モックStudyを作成
        mock_trial = MagicMock()
        mock_trial.number = 0
        mock_trial.value = 0.95
        mock_trial.params = {"learning_rate": 0.001}
        mock_trial.state.name = "COMPLETE"
        mock_trial.datetime_start = None
        mock_trial.datetime_complete = None

        mock_study = MagicMock()
        mock_study.study_name = "test_study"
        mock_study.trials = [mock_trial]

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter.export(
                best_params={"learning_rate": 0.001},
                best_value=0.95,
                study=mock_study,
                output_path=tmpdir,
            )

            # best_params.jsonの確認
            best_params_file = Path(tmpdir) / "best_params.json"
            assert best_params_file.exists()

            with open(best_params_file) as f:
                data = json.load(f)

            assert data["study_name"] == "test_study"
            assert data["best_value"] == 0.95
            assert data["best_params"]["learning_rate"] == 0.001
            assert data["n_trials"] == 1

    def test_export_creates_trials_history_json(self) -> None:
        """trials_history.jsonが作成されることをテスト."""
        exporter = JsonResultExporter()

        mock_trial1 = MagicMock()
        mock_trial1.number = 0
        mock_trial1.value = 0.90
        mock_trial1.params = {"learning_rate": 0.01}
        mock_trial1.state.name = "COMPLETE"
        mock_trial1.datetime_start = None
        mock_trial1.datetime_complete = None

        mock_trial2 = MagicMock()
        mock_trial2.number = 1
        mock_trial2.value = 0.95
        mock_trial2.params = {"learning_rate": 0.001}
        mock_trial2.state.name = "COMPLETE"
        mock_trial2.datetime_start = None
        mock_trial2.datetime_complete = None

        mock_study = MagicMock()
        mock_study.study_name = "test_study"
        mock_study.trials = [mock_trial1, mock_trial2]

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter.export(
                best_params={"learning_rate": 0.001},
                best_value=0.95,
                study=mock_study,
                output_path=tmpdir,
            )

            trials_file = Path(tmpdir) / "trials_history.json"
            assert trials_file.exists()

            with open(trials_file) as f:
                data = json.load(f)

            assert len(data) == 2
            assert data[0]["number"] == 0
            assert data[1]["number"] == 1


class TestConfigExporter:
    """ConfigExporterのテスト."""

    def test_export_creates_optimized_config_py(self) -> None:
        """optimized_config.pyが作成されることをテスト."""
        base_config = {
            "model_name": "resnet18",
            "num_classes": 10,
            "epochs": 50,
        }
        exporter = ConfigExporter(base_config)

        mock_study = MagicMock()
        mock_study.study_name = "test_study"
        mock_study.trials = []

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter.export(
                best_params={"learning_rate": 0.001, "optimizer": "Adam"},
                best_value=0.95,
                study=mock_study,
                output_path=tmpdir,
            )

            config_file = Path(tmpdir) / "optimized_config.py"
            assert config_file.exists()

            content = config_file.read_text(encoding="utf-8")

            # 最適化されたパラメータが含まれていることを確認
            assert "learning_rate = 0.001" in content
            assert "optimizer = 'Adam'" in content

            # ベース設定も含まれていることを確認
            assert "model_name = 'resnet18'" in content
            assert "num_classes = 10" in content
