"""目的関数実装（SRP: 単一責任原則）."""

from typing import Any

import optuna
from torch.utils.data import DataLoader

from pochitrain.config.pochi_config import PochiConfig
from pochitrain.optimization.interfaces import IObjectiveFunction, IParamSuggestor


class ClassificationObjective(IObjectiveFunction):
    """画像分類タスク用の目的関数.

    PochiTrainerを使用して訓練を実行し、検証精度を返す。
    """

    def __init__(
        self,
        base_config: PochiConfig,
        param_suggestor: IParamSuggestor,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optuna_epochs: int = 10,
        device: str = "cuda",
    ) -> None:
        """初期化.

        Args:
            base_config: ベース設定（PochiConfig dataclass）
            param_suggestor: パラメータサジェスター
            train_loader: 訓練データローダー
            val_loader: 検証データローダー
            optuna_epochs: 最適化時のエポック数（短めに設定）
            device: デバイス
        """
        self._base_config = base_config
        self._param_suggestor = param_suggestor
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._optuna_epochs = optuna_epochs
        self._device = device

    def __call__(self, trial: optuna.Trial) -> float:
        """目的関数を実行する.

        Args:
            trial: Optuna trial オブジェクト

        Returns:
            検証精度（最大化対象）
        """
        # 遅延インポート（循環参照回避）
        from pochitrain import PochiTrainer

        # パラメータをサジェスト
        suggested_params = self._param_suggestor.suggest(trial)

        # トレーナーを作成（ワークスペースは作成しない）
        trainer = PochiTrainer(
            model_name=self._base_config.model_name,
            num_classes=self._base_config.num_classes,
            device=self._device,
            pretrained=self._base_config.pretrained,
            create_workspace=False,  # Optuna最適化中はワークスペース不要
        )

        # 訓練設定をセットアップ
        trainer.setup_training(
            learning_rate=suggested_params.get(
                "learning_rate",
                self._base_config.learning_rate,
            ),
            optimizer_name=suggested_params.get(
                "optimizer",
                self._base_config.optimizer,
            ),
            scheduler_name=suggested_params.get(
                "scheduler",
                self._base_config.scheduler,
            ),
            scheduler_params=suggested_params.get(
                "scheduler_params",
                self._base_config.scheduler_params,
            ),
            enable_layer_wise_lr=suggested_params.get("enable_layer_wise_lr", False),
            layer_wise_lr_config=suggested_params.get("layer_wise_lr_config"),
        )

        # 訓練実行（短いエポック数）
        best_accuracy = self._train_and_evaluate(trainer, trial)

        return best_accuracy

    def _train_and_evaluate(
        self,
        trainer: Any,
        trial: optuna.Trial,
    ) -> float:
        """訓練を実行し、最良の精度を返す.

        Args:
            trainer: PochiTrainerインスタンス
            trial: Optuna trial（pruning用）

        Returns:
            最良の検証精度
        """
        best_accuracy = 0.0

        for epoch in range(1, self._optuna_epochs + 1):
            # 1エポック訓練
            trainer.train_one_epoch(epoch=epoch, train_loader=self._train_loader)

            # 検証
            val_metrics = trainer.validate(self._val_loader)
            accuracy = val_metrics.get("val_accuracy", 0.0)

            # 最良精度を更新
            if accuracy > best_accuracy:
                best_accuracy = accuracy

            # Pruning: 中間結果を報告
            trial.report(accuracy, epoch)

            # Pruningチェック
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_accuracy
