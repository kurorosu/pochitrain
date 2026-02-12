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
    """Objective テスト用の簡易 Trainer."""

    last_init_kwargs: dict[str, Any] = {}
    last_setup_kwargs: dict[str, Any] = {}
    validate_values: list[dict[str, float]] = []

    @classmethod
    def reset(cls) -> None:
        """テスト間の状態を初期化する."""
        cls.last_init_kwargs = {}
        cls.last_setup_kwargs = {}
        cls.validate_values = []

    def __init__(self, **kwargs: Any) -> None:
        """初期化時引数を保存する.

        Args:
            **kwargs: 初期化引数.
        """
        type(self).last_init_kwargs = dict(kwargs)

    def setup_training(self, **kwargs: Any) -> None:
        """学習設定引数を保存する.

        Args:
            **kwargs: 学習設定引数.
        """
        type(self).last_setup_kwargs = dict(kwargs)

    def train_epoch(self, _train_loader: Any) -> None:
        """1 エポック学習を模擬する.

        Args:
            _train_loader: 未使用.
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

    def test_call_creates_trainer_with_config_attributes(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """PochiConfig 由来の属性で Trainer が生成されることを検証する."""
        monkeypatch.setattr(pochitrain, "PochiTrainer", FakeTrainer)
        FakeTrainer.validate_values = [{"val_accuracy": 0.85}]

        objective = _build_objective(StaticParamSuggestor({"learning_rate": 0.01}))
        trial = DummyTrial()

        objective(trial)

        assert FakeTrainer.last_init_kwargs == {
            "model_name": "resnet18",
            "num_classes": 5,
            "device": "cpu",
            "pretrained": True,
            "create_workspace": False,
        }

    def test_call_uses_suggested_params_over_config(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """提案パラメータが優先されることを検証する."""
        monkeypatch.setattr(pochitrain, "PochiTrainer", FakeTrainer)
        FakeTrainer.validate_values = [{"val_accuracy": 0.90}]

        objective = _build_objective(
            StaticParamSuggestor(
                {
                    "learning_rate": 0.05,
                    "optimizer": "SGD",
                }
            )
        )

        objective(DummyTrial())

        assert FakeTrainer.last_setup_kwargs == {
            "learning_rate": 0.05,
            "optimizer_name": "SGD",
            "scheduler_name": None,
            "scheduler_params": None,
            "enable_layer_wise_lr": False,
            "layer_wise_lr_config": None,
        }

    def test_call_falls_back_to_config_when_not_suggested(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """未提案パラメータはベース設定を使うことを検証する."""
        monkeypatch.setattr(pochitrain, "PochiTrainer", FakeTrainer)
        FakeTrainer.validate_values = [{"val_accuracy": 0.80}]

        base_config = _create_base_config()
        base_config.scheduler = "StepLR"
        base_config.scheduler_params = {"step_size": 10}
        objective = ClassificationObjective(
            base_config=base_config,
            param_suggestor=StaticParamSuggestor({}),
            train_loader=[],
            val_loader=[],
            optuna_epochs=1,
            device="cpu",
        )

        objective(DummyTrial())

        assert FakeTrainer.last_setup_kwargs == {
            "learning_rate": 0.001,
            "optimizer_name": "Adam",
            "scheduler_name": "StepLR",
            "scheduler_params": {"step_size": 10},
            "enable_layer_wise_lr": False,
            "layer_wise_lr_config": None,
        }

    def test_call_returns_best_accuracy(
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
