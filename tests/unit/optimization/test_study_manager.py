"""StudyManager のユニットテスト."""

import optuna
import pytest

from pochitrain.optimization.interfaces import IObjectiveFunction
from pochitrain.optimization.study_manager import OptunaStudyManager


class FixedObjective(IObjectiveFunction):
    """一定の値を返すテスト用 objective."""

    def __init__(self, value: float = 0.5) -> None:
        """初期化する.

        Args:
            value: 常に返す評価値.
        """
        self._value = value

    def __call__(self, _trial: optuna.Trial) -> float:
        """評価値を返す.

        Args:
            _trial: Optuna trial.

        Returns:
            固定の評価値.
        """
        return self._value


class TestOptunaStudyManager:
    """OptunaStudyManager のテスト."""

    def test_create_study_default(self) -> None:
        """デフォルト設定で Study を作成できることを検証する."""
        manager = OptunaStudyManager()

        study = manager.create_study(study_name="test_study_default")

        assert study.study_name == "test_study_default"
        assert study.direction == optuna.study.StudyDirection.MAXIMIZE
        assert isinstance(study.sampler, optuna.samplers.TPESampler)
        assert manager.get_study() is study

    def test_create_study_with_sampler(self) -> None:
        """サンプラー指定で Study を作成できることを検証する."""
        manager = OptunaStudyManager()

        study = manager.create_study(
            study_name="test_study_sampler",
            sampler="RandomSampler",
        )

        assert isinstance(study.sampler, optuna.samplers.RandomSampler)

    def test_create_study_with_pruner(self) -> None:
        """プルーナー指定で Study を作成できることを検証する."""
        manager = OptunaStudyManager()

        study = manager.create_study(
            study_name="test_study_pruner",
            pruner="MedianPruner",
        )

        assert isinstance(study.pruner, optuna.pruners.MedianPruner)

    def test_optimize_without_study_raises_error(self) -> None:
        """Study 未作成で optimize を呼ぶと例外になることを検証する."""
        manager = OptunaStudyManager()

        with pytest.raises(RuntimeError, match="Study not created"):
            manager.optimize(FixedObjective(), n_trials=1)

    def test_get_best_params_without_study_raises_error(self) -> None:
        """Study 未作成で best params を取ると例外になることを検証する."""
        manager = OptunaStudyManager()

        with pytest.raises(RuntimeError, match="Study not created"):
            manager.get_best_params()

    def test_get_best_value_without_study_raises_error(self) -> None:
        """Study 未作成で best value を取ると例外になることを検証する."""
        manager = OptunaStudyManager()

        with pytest.raises(RuntimeError, match="Study not created"):
            manager.get_best_value()

    def test_optimize_updates_best_params_and_value(self) -> None:
        """optimize 実行後に best params と best value を取得できることを検証する."""

        class LearningRateObjective(IObjectiveFunction):
            """learning_rate を返す objective."""

            def __call__(self, trial: optuna.Trial) -> float:
                """試行値を返す.

                Args:
                    trial: Optuna trial.

                Returns:
                    learning_rate の値.
                """
                learning_rate = trial.suggest_float(
                    "learning_rate",
                    1e-5,
                    1e-1,
                    log=True,
                )
                return float(learning_rate)

        manager = OptunaStudyManager()
        manager.create_study(study_name="test_study_optimize")

        manager.optimize(LearningRateObjective(), n_trials=3)

        best_params = manager.get_best_params()
        best_value = manager.get_best_value()
        study = manager.get_study()

        assert study is not None
        assert len(study.trials) == 3
        assert "learning_rate" in best_params
        assert isinstance(best_value, float)

    def test_unknown_sampler_raises_error(self) -> None:
        """未知のサンプラー名では例外になることを検証する."""
        manager = OptunaStudyManager()

        with pytest.raises(ValueError, match="Unknown sampler"):
            manager._create_sampler("UnknownSampler")

    def test_unknown_pruner_raises_error(self) -> None:
        """未知のプルーナー名では例外になることを検証する."""
        manager = OptunaStudyManager()

        with pytest.raises(ValueError, match="Unknown pruner"):
            manager._create_pruner("UnknownPruner")
