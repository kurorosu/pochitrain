"""ClassificationObjective のユニットテスト."""

from typing import Any

import optuna
import pytest

import pochitrain
from pochitrain.config.pochi_config import PochiConfig
from pochitrain.optimization.interfaces import IParamSuggestor
from pochitrain.optimization.objective import ClassificationObjective


class StaticParamSuggestor(IParamSuggestor):
    """固定パラメータを返す簡易 suggestor."""

    def __init__(self, params: dict[str, Any]) -> None:
        """初期化する.

        Args:
            params: 常に返すパラメータ.
        """
        self._params = params

    def suggest(self, _trial: Any) -> dict[str, Any]:
        """固定パラメータを返す.

        Args:
            _trial: 未使用の trial オブジェクト.

        Returns:
            固定パラメータ.
        """
        return dict(self._params)


class DummyTrial:
    """report と should_prune だけを持つ簡易 trial."""

    def __init__(self, prune_steps: set[int] | None = None) -> None:
        """初期化する.

        Args:
            prune_steps: prune を返す step の集合.
        """
        self._prune_steps = prune_steps or set()
        self._current_step = 0
        self.reported: list[tuple[float, int]] = []

    def report(self, value: float, step: int) -> None:
        """中間値を記録する.

        Args:
            value: 報告値.
            step: エポック番号.
        """
        self.reported.append((value, step))
        self._current_step = step

    def should_prune(self) -> bool:
        """現在 step が prune 対象かを返す.

        Returns:
            prune する場合は True.
        """
        return self._current_step in self._prune_steps


class FakeTrainer:
    """Objective テスト用の簡易 Trainer.

    観測可能な出力 (validate の戻り値) のみを制御し,
    内部呼び出しの引数検証は行わない.
    """

    validate_values: list[dict[str, float]] = []

    @classmethod
    def reset(cls) -> None:
        """テスト間の状態を初期化する."""
        cls.validate_values = []

    def __init__(self, **_kwargs: Any) -> None:
        """初期化する.

        Args:
            **_kwargs: 未使用 (PochiTrainer 互換シグネチャ).
        """

    def setup_training(self, **_kwargs: Any) -> None:
        """学習を設定する.

        Args:
            **_kwargs: 未使用 (PochiTrainer 互換シグネチャ).
        """

    def train_one_epoch(self, **_kwargs: Any) -> None:
        """1 エポック学習を模擬する.

        Args:
            **_kwargs: 未使用.
        """

    def validate(self, _val_loader: Any) -> dict[str, float]:
        """事前に設定した検証値を返す.

        Args:
            _val_loader: 未使用.

        Returns:
            検証メトリクス.
        """
        if type(self).validate_values:
            return type(self).validate_values.pop(0)
        return {"val_accuracy": 0.0}


def _create_base_config() -> PochiConfig:
    """テスト用の最小 PochiConfig を生成する.

    Returns:
        テスト用設定オブジェクト.
    """
    return PochiConfig(
        model_name="resnet18",
        num_classes=5,
        device="cpu",
        epochs=10,
        batch_size=32,
        learning_rate=0.001,
        optimizer="Adam",
        train_data_root="data/train",
        train_transform=object(),
        val_transform=object(),
        enable_layer_wise_lr=False,
    )


def _build_objective(
    suggestor: StaticParamSuggestor,
    optuna_epochs: int = 1,
) -> ClassificationObjective:
    """ClassificationObjective を生成する.

    Args:
        suggestor: パラメータ suggestor.
        optuna_epochs: 最適化時エポック数.

    Returns:
        ClassificationObjective インスタンス.
    """
    return ClassificationObjective(
        base_config=_create_base_config(),
        param_suggestor=suggestor,
        train_loader=[],
        val_loader=[],
        optuna_epochs=optuna_epochs,
        device="cpu",
    )


class TestClassificationObjective:
    """ClassificationObjective のテスト."""

    def setup_method(self) -> None:
        """各テスト前に FakeTrainer 状態を初期化する."""
        FakeTrainer.reset()

    def test_call_returns_accuracy_from_single_epoch(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """1 エポック実行時に検証精度がそのまま返ることを検証する."""
        monkeypatch.setattr(pochitrain, "PochiTrainer", FakeTrainer)
        FakeTrainer.validate_values = [{"val_accuracy": 0.85}]

        objective = _build_objective(StaticParamSuggestor({"learning_rate": 0.01}))
        trial = DummyTrial()

        result = objective(trial)

        assert result == pytest.approx(0.85)
        assert trial.reported == [(0.85, 1)]

    def test_call_returns_best_accuracy_across_epochs(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """複数エポックで最大精度が返ることを検証する."""
        monkeypatch.setattr(pochitrain, "PochiTrainer", FakeTrainer)
        FakeTrainer.validate_values = [
            {"val_accuracy": 0.70},
            {"val_accuracy": 0.85},
            {"val_accuracy": 0.80},
        ]

        objective = _build_objective(StaticParamSuggestor({}), optuna_epochs=3)

        result = objective(DummyTrial())

        assert result == pytest.approx(0.85)

    def test_call_reports_each_epoch_accuracy_to_trial(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """各エポックの精度が trial に報告されることを検証する."""
        monkeypatch.setattr(pochitrain, "PochiTrainer", FakeTrainer)
        FakeTrainer.validate_values = [
            {"val_accuracy": 0.60},
            {"val_accuracy": 0.75},
        ]

        objective = _build_objective(StaticParamSuggestor({}), optuna_epochs=2)
        trial = DummyTrial()

        objective(trial)

        assert trial.reported == [(0.60, 1), (0.75, 2)]

    def test_call_raises_trial_pruned_when_should_prune(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """trial が prune 判定した場合に TrialPruned が送出されることを検証する."""
        monkeypatch.setattr(pochitrain, "PochiTrainer", FakeTrainer)
        FakeTrainer.validate_values = [
            {"val_accuracy": 0.71},
            {"val_accuracy": 0.72},
        ]

        objective = _build_objective(StaticParamSuggestor({}), optuna_epochs=2)
        trial = DummyTrial(prune_steps={2})

        with pytest.raises(optuna.TrialPruned):
            objective(trial)

        assert trial.reported == [(0.71, 1), (0.72, 2)]

    def test_call_returns_zero_when_no_accuracy_in_metrics(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """validate が val_accuracy を含まない場合に 0.0 が返ることを検証する."""
        monkeypatch.setattr(pochitrain, "PochiTrainer", FakeTrainer)
        FakeTrainer.validate_values = [{"val_loss": 1.5}]

        objective = _build_objective(StaticParamSuggestor({}))

        result = objective(DummyTrial())

        assert result == pytest.approx(0.0)
