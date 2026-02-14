"""ResultExporter のユニットテスト."""

import json
from pathlib import Path

import optuna
import pytest
from torchvision import transforms

from pochitrain.config.pochi_config import PochiConfig
from pochitrain.optimization.result_exporter import (
    ConfigExporter,
    JsonResultExporter,
)


def _create_test_study() -> optuna.Study:
    """エクスポート検証用の Study を生成する.

    Returns:
        最適化済みの Optuna Study.
    """

    def objective(trial: optuna.Trial) -> float:
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        optimizer = trial.suggest_categorical("optimizer", ["SGD", "Adam"])
        optimizer_bonus = 0.02 if optimizer == "Adam" else 0.0
        return float(0.9 + optimizer_bonus - abs(learning_rate - 0.001))

    study = optuna.create_study(study_name="test_study", direction="maximize")
    study.optimize(objective, n_trials=2, show_progress_bar=False)
    return study


class TestJsonResultExporter:
    """JsonResultExporter のテスト."""

    def test_export_creates_best_params_json(self, tmp_path: Path) -> None:
        """best_params.json を出力できることを検証する."""
        exporter = JsonResultExporter()
        study = _create_test_study()

        exporter.export(
            best_params=study.best_params,
            best_value=float(study.best_value),
            study=study,
            output_path=str(tmp_path),
        )

        best_params_file = tmp_path / "best_params.json"
        assert best_params_file.exists()

        data = json.loads(best_params_file.read_text(encoding="utf-8"))
        assert data["study_name"] == "test_study"
        assert data["best_value"] == pytest.approx(float(study.best_value))
        assert data["n_trials"] == len(study.trials)

    def test_export_creates_trials_history_json(self, tmp_path: Path) -> None:
        """trials_history.json を出力できることを検証する."""
        exporter = JsonResultExporter()
        study = _create_test_study()

        exporter.export(
            best_params=study.best_params,
            best_value=float(study.best_value),
            study=study,
            output_path=str(tmp_path),
        )

        trials_file = tmp_path / "trials_history.json"
        assert trials_file.exists()

        data = json.loads(trials_file.read_text(encoding="utf-8"))
        assert len(data) == len(study.trials)
        assert {"number", "value", "params", "state"}.issubset(data[0])


class TestConfigExporter:
    """ConfigExporter のテスト."""

    def test_export_creates_optimized_config_py(self, tmp_path: Path) -> None:
        """optimized_config.py を生成できることを検証する."""
        base_config = PochiConfig(
            model_name="resnet18",
            num_classes=10,
            device="cpu",
            epochs=50,
            batch_size=32,
            learning_rate=0.001,
            optimizer="Adam",
            train_data_root="data/train",
            train_transform=transforms.Compose([transforms.ToTensor()]),
            val_transform=transforms.Compose([transforms.ToTensor()]),
            enable_layer_wise_lr=False,
        )
        exporter = ConfigExporter(base_config)
        study = _create_test_study()

        exporter.export(
            best_params=study.best_params,
            best_value=float(study.best_value),
            study=study,
            output_path=str(tmp_path),
        )

        config_file = tmp_path / "optimized_config.py"
        assert config_file.exists()

        content = config_file.read_text(encoding="utf-8")
        assert "learning_rate =" in content
        assert "optimizer =" in content
        assert "model_name = 'resnet18'" in content
        assert "num_classes = 10" in content
        assert "train_transform = transforms.Compose(" in content
        assert "val_transform = transforms.Compose(" in content
