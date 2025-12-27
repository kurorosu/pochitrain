"""Optuna Study管理実装（SRP: 単一責任原則）."""

from typing import Any

import optuna

from pochitrain.optimization.interfaces import IObjectiveFunction, IStudyManager


class OptunaStudyManager(IStudyManager):
    """Optuna Studyを管理するクラス.

    Studyの作成・実行・結果取得を担当する。
    """

    def __init__(self, storage: str | None = None) -> None:
        """初期化.

        Args:
            storage: Optuna storage URL（オプション）
                例: "sqlite:///optuna_study.db"
        """
        self._storage = storage
        self._study: optuna.Study | None = None

    def create_study(
        self,
        study_name: str,
        direction: str = "maximize",
        sampler: str = "TPESampler",
        pruner: str | None = None,
    ) -> optuna.Study:
        """Optuna Studyを作成する.

        Args:
            study_name: Study名
            direction: 最適化方向 ("maximize" or "minimize")
            sampler: サンプラー名
            pruner: プルーナー名（オプション）

        Returns:
            作成されたStudyオブジェクト
        """
        sampler_instance = self._create_sampler(sampler)
        pruner_instance = self._create_pruner(pruner) if pruner else None

        self._study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=sampler_instance,
            pruner=pruner_instance,
            storage=self._storage,
            load_if_exists=True,
        )

        return self._study

    def optimize(
        self,
        objective: IObjectiveFunction,
        n_trials: int,
        n_jobs: int = 1,
    ) -> None:
        """最適化を実行する.

        Args:
            objective: 目的関数
            n_trials: 試行回数
            n_jobs: 並列ジョブ数

        Raises:
            RuntimeError: Studyが作成されていない場合
        """
        if self._study is None:
            msg = "Study not created. Call create_study() first."
            raise RuntimeError(msg)

        self._study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            show_progress_bar=True,
        )

    def get_best_params(self) -> dict[str, Any]:
        """最適なパラメータを取得する.

        Returns:
            最適なハイパーパラメータの辞書

        Raises:
            RuntimeError: Studyが作成されていない場合
        """
        if self._study is None:
            msg = "Study not created. Call create_study() first."
            raise RuntimeError(msg)

        return self._study.best_params

    def get_best_value(self) -> float:
        """最適な目的関数値を取得する.

        Returns:
            最適な目的関数値

        Raises:
            RuntimeError: Studyが作成されていない場合
        """
        if self._study is None:
            msg = "Study not created. Call create_study() first."
            raise RuntimeError(msg)

        return self._study.best_value

    def get_study(self) -> optuna.Study | None:
        """Studyオブジェクトを取得する.

        Returns:
            Studyオブジェクト（未作成の場合はNone）
        """
        return self._study

    def _create_sampler(self, sampler_name: str) -> optuna.samplers.BaseSampler:
        """サンプラーを作成する.

        Args:
            sampler_name: サンプラー名

        Returns:
            サンプラーインスタンス
        """
        samplers = {
            "TPESampler": optuna.samplers.TPESampler,
            "RandomSampler": optuna.samplers.RandomSampler,
            "CmaEsSampler": optuna.samplers.CmaEsSampler,
            "GridSampler": optuna.samplers.GridSampler,
        }

        if sampler_name not in samplers:
            msg = f"Unknown sampler: {sampler_name}. Available: {list(samplers.keys())}"
            raise ValueError(msg)

        return samplers[sampler_name]()

    def _create_pruner(self, pruner_name: str) -> optuna.pruners.BasePruner:
        """プルーナーを作成する.

        Args:
            pruner_name: プルーナー名

        Returns:
            プルーナーインスタンス
        """
        pruners = {
            "MedianPruner": optuna.pruners.MedianPruner,
            "PercentilePruner": optuna.pruners.PercentilePruner,
            "SuccessiveHalvingPruner": optuna.pruners.SuccessiveHalvingPruner,
            "HyperbandPruner": optuna.pruners.HyperbandPruner,
            "NopPruner": optuna.pruners.NopPruner,
        }

        if pruner_name not in pruners:
            msg = f"Unknown pruner: {pruner_name}. Available: {list(pruners.keys())}"
            raise ValueError(msg)

        return pruners[pruner_name]()
