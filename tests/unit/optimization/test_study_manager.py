"""StudyManagerのユニットテスト."""

from unittest.mock import MagicMock, patch

import pytest

from pochitrain.optimization.study_manager import OptunaStudyManager


class TestOptunaStudyManager:
    """OptunaStudyManagerのテスト."""

    @patch("pochitrain.optimization.study_manager.optuna")
    def test_create_study_default(self, mock_optuna: MagicMock) -> None:
        """デフォルト設定でStudy作成をテスト."""
        mock_study = MagicMock()
        mock_optuna.create_study.return_value = mock_study
        mock_optuna.samplers.TPESampler.return_value = MagicMock()

        manager = OptunaStudyManager()
        study = manager.create_study(study_name="test_study")

        assert study == mock_study
        mock_optuna.create_study.assert_called_once()

    @patch("pochitrain.optimization.study_manager.optuna")
    def test_create_study_with_sampler(self, mock_optuna: MagicMock) -> None:
        """サンプラー指定でStudy作成をテスト."""
        mock_study = MagicMock()
        mock_optuna.create_study.return_value = mock_study
        mock_optuna.samplers.RandomSampler.return_value = MagicMock()

        manager = OptunaStudyManager()
        study = manager.create_study(
            study_name="test_study",
            sampler="RandomSampler",
        )

        mock_optuna.samplers.RandomSampler.assert_called_once()

    @patch("pochitrain.optimization.study_manager.optuna")
    def test_create_study_with_pruner(self, mock_optuna: MagicMock) -> None:
        """プルーナー指定でStudy作成をテスト."""
        mock_study = MagicMock()
        mock_optuna.create_study.return_value = mock_study
        mock_optuna.samplers.TPESampler.return_value = MagicMock()
        mock_optuna.pruners.MedianPruner.return_value = MagicMock()

        manager = OptunaStudyManager()
        study = manager.create_study(
            study_name="test_study",
            pruner="MedianPruner",
        )

        mock_optuna.pruners.MedianPruner.assert_called_once()

    def test_optimize_without_study_raises_error(self) -> None:
        """Study未作成で最適化実行するとエラーをテスト."""
        manager = OptunaStudyManager()
        objective = MagicMock()

        with pytest.raises(RuntimeError, match="Study not created"):
            manager.optimize(objective, n_trials=10)

    def test_get_best_params_without_study_raises_error(self) -> None:
        """Study未作成でベストパラメータ取得するとエラーをテスト."""
        manager = OptunaStudyManager()

        with pytest.raises(RuntimeError, match="Study not created"):
            manager.get_best_params()

    def test_get_best_value_without_study_raises_error(self) -> None:
        """Study未作成でベスト値取得するとエラーをテスト."""
        manager = OptunaStudyManager()

        with pytest.raises(RuntimeError, match="Study not created"):
            manager.get_best_value()

    @patch("pochitrain.optimization.study_manager.optuna")
    def test_get_best_params(self, mock_optuna: MagicMock) -> None:
        """ベストパラメータ取得をテスト."""
        mock_study = MagicMock()
        mock_study.best_params = {"learning_rate": 0.001}
        mock_optuna.create_study.return_value = mock_study
        mock_optuna.samplers.TPESampler.return_value = MagicMock()

        manager = OptunaStudyManager()
        manager.create_study(study_name="test_study")

        best_params = manager.get_best_params()

        assert best_params == {"learning_rate": 0.001}

    @patch("pochitrain.optimization.study_manager.optuna")
    def test_get_best_value(self, mock_optuna: MagicMock) -> None:
        """ベスト値取得をテスト."""
        mock_study = MagicMock()
        mock_study.best_value = 0.95
        mock_optuna.create_study.return_value = mock_study
        mock_optuna.samplers.TPESampler.return_value = MagicMock()

        manager = OptunaStudyManager()
        manager.create_study(study_name="test_study")

        best_value = manager.get_best_value()

        assert best_value == 0.95

    def test_unknown_sampler_raises_error(self) -> None:
        """未知のサンプラー名でエラーが発生することをテスト."""
        manager = OptunaStudyManager()

        with pytest.raises(ValueError, match="Unknown sampler"):
            manager._create_sampler("UnknownSampler")

    def test_unknown_pruner_raises_error(self) -> None:
        """未知のプルーナー名でエラーが発生することをテスト."""
        manager = OptunaStudyManager()

        with pytest.raises(ValueError, match="Unknown pruner"):
            manager._create_pruner("UnknownPruner")
