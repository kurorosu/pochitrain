"""ClassificationObjectiveのユニットテスト."""

from unittest.mock import MagicMock, patch

from pochitrain.config.pochi_config import PochiConfig
from pochitrain.optimization.objective import ClassificationObjective


def _create_base_config() -> PochiConfig:
    """テスト用PochiConfigを作成."""
    return PochiConfig(
        model_name="resnet18",
        num_classes=5,
        device="cpu",
        epochs=10,
        batch_size=32,
        learning_rate=0.001,
        optimizer="Adam",
        train_data_root="data/train",
        train_transform=MagicMock(),
        val_transform=MagicMock(),
        enable_layer_wise_lr=False,
    )


class TestClassificationObjective:
    """ClassificationObjectiveのテスト."""

    @patch("pochitrain.PochiTrainer", autospec=True)
    def test_call_creates_trainer_with_config_attributes(
        self, mock_trainer_cls: MagicMock
    ) -> None:
        """PochiConfigの属性でTrainerが作成されることをテスト."""
        config = _create_base_config()
        mock_suggestor = MagicMock()
        mock_suggestor.suggest.return_value = {"learning_rate": 0.01}

        mock_trainer = MagicMock()
        mock_trainer.validate.return_value = {"val_accuracy": 0.85}
        mock_trainer_cls.return_value = mock_trainer

        objective = ClassificationObjective(
            base_config=config,
            param_suggestor=mock_suggestor,
            train_loader=MagicMock(),
            val_loader=MagicMock(),
            optuna_epochs=1,
            device="cpu",
        )

        mock_trial = MagicMock()
        mock_trial.should_prune.return_value = False
        objective(mock_trial)

        mock_trainer_cls.assert_called_once_with(
            model_name="resnet18",
            num_classes=5,
            device="cpu",
            pretrained=True,
            create_workspace=False,
        )

    @patch("pochitrain.PochiTrainer", autospec=True)
    def test_call_uses_suggested_params_over_config(
        self, mock_trainer_cls: MagicMock
    ) -> None:
        """サジェストされたパラメータがベース設定より優先されることをテスト."""
        config = _create_base_config()
        mock_suggestor = MagicMock()
        mock_suggestor.suggest.return_value = {
            "learning_rate": 0.05,
            "optimizer": "SGD",
        }

        mock_trainer = MagicMock()
        mock_trainer.validate.return_value = {"val_accuracy": 0.90}
        mock_trainer_cls.return_value = mock_trainer

        objective = ClassificationObjective(
            base_config=config,
            param_suggestor=mock_suggestor,
            train_loader=MagicMock(),
            val_loader=MagicMock(),
            optuna_epochs=1,
            device="cpu",
        )

        mock_trial = MagicMock()
        mock_trial.should_prune.return_value = False
        objective(mock_trial)

        mock_trainer.setup_training.assert_called_once_with(
            learning_rate=0.05,
            optimizer_name="SGD",
            scheduler_name=None,
            scheduler_params=None,
            enable_layer_wise_lr=False,
            layer_wise_lr_config=None,
        )

    @patch("pochitrain.PochiTrainer", autospec=True)
    def test_call_falls_back_to_config_when_not_suggested(
        self, mock_trainer_cls: MagicMock
    ) -> None:
        """サジェストされないパラメータはベース設定が使われることをテスト."""
        config = _create_base_config()
        config.scheduler = "StepLR"
        config.scheduler_params = {"step_size": 10}

        mock_suggestor = MagicMock()
        mock_suggestor.suggest.return_value = {}

        mock_trainer = MagicMock()
        mock_trainer.validate.return_value = {"val_accuracy": 0.80}
        mock_trainer_cls.return_value = mock_trainer

        objective = ClassificationObjective(
            base_config=config,
            param_suggestor=mock_suggestor,
            train_loader=MagicMock(),
            val_loader=MagicMock(),
            optuna_epochs=1,
            device="cpu",
        )

        mock_trial = MagicMock()
        mock_trial.should_prune.return_value = False
        objective(mock_trial)

        mock_trainer.setup_training.assert_called_once_with(
            learning_rate=0.001,
            optimizer_name="Adam",
            scheduler_name="StepLR",
            scheduler_params={"step_size": 10},
            enable_layer_wise_lr=False,
            layer_wise_lr_config=None,
        )

    @patch("pochitrain.PochiTrainer", autospec=True)
    def test_call_returns_best_accuracy(self, mock_trainer_cls: MagicMock) -> None:
        """最良の検証精度が返されることをテスト."""
        config = _create_base_config()
        mock_suggestor = MagicMock()
        mock_suggestor.suggest.return_value = {}

        mock_trainer = MagicMock()
        mock_trainer.validate.side_effect = [
            {"val_accuracy": 0.70},
            {"val_accuracy": 0.85},
            {"val_accuracy": 0.80},
        ]
        mock_trainer_cls.return_value = mock_trainer

        objective = ClassificationObjective(
            base_config=config,
            param_suggestor=mock_suggestor,
            train_loader=MagicMock(),
            val_loader=MagicMock(),
            optuna_epochs=3,
            device="cpu",
        )

        mock_trial = MagicMock()
        mock_trial.should_prune.return_value = False
        result = objective(mock_trial)

        assert result == 0.85
