"""抽象インターフェース定義（DIP: 依存性逆転原則）."""

from abc import ABC, abstractmethod
from typing import Any


class IParamSuggestor(ABC):
    """ハイパーパラメータサジェスターのインターフェース.

    Optunaのtrialからハイパーパラメータを提案する責務を持つ。
    """

    @abstractmethod
    def suggest(self, trial: Any) -> dict[str, Any]:
        """Optuna trialからハイパーパラメータを提案する.

        Args:
            trial: Optuna trial オブジェクト

        Returns:
            サジェストされたハイパーパラメータの辞書
        """
        pass


class IObjectiveFunction(ABC):
    """目的関数のインターフェース.

    Optuna最適化の目的関数を定義する責務を持つ。
    """

    @abstractmethod
    def __call__(self, trial: Any) -> float:
        """目的関数を実行する.

        Args:
            trial: Optuna trial オブジェクト

        Returns:
            最適化対象のスコア（精度など）
        """
        pass


class IStudyManager(ABC):
    """Optuna Study管理のインターフェース.

    Studyの作成・実行・結果取得の責務を持つ。
    """

    @abstractmethod
    def create_study(
        self,
        study_name: str,
        direction: str = "maximize",
        sampler: str = "TPESampler",
        pruner: str | None = None,
    ) -> Any:
        """Optuna Studyを作成する.

        Args:
            study_name: Study名
            direction: 最適化方向 ("maximize" or "minimize")
            sampler: サンプラー名
            pruner: プルーナー名（オプション）

        Returns:
            作成されたStudyオブジェクト
        """
        pass

    @abstractmethod
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
        """
        pass

    @abstractmethod
    def get_best_params(self) -> dict[str, Any]:
        """最適なパラメータを取得する.

        Returns:
            最適なハイパーパラメータの辞書
        """
        pass

    @abstractmethod
    def get_best_value(self) -> float:
        """最適な目的関数値を取得する.

        Returns:
            最適な目的関数値
        """
        pass


class IResultExporter(ABC):
    """最適化結果エクスポーターのインターフェース.

    最適化結果の保存・出力の責務を持つ。
    """

    @abstractmethod
    def export(
        self,
        best_params: dict[str, Any],
        best_value: float,
        study: Any,
        output_path: str,
    ) -> None:
        """最適化結果をエクスポートする.

        Args:
            best_params: 最適なパラメータ
            best_value: 最適な目的関数値
            study: Optuna Studyオブジェクト
            output_path: 出力先パス
        """
        pass
