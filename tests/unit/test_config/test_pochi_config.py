"""PochiConfig のテスト."""

from typing import Any

import pytest
from pydantic import ValidationError

from pochitrain.config.pochi_config import PochiConfig
from pochitrain.utils import ConfigLoader


def _build_minimum_config() -> dict[str, Any]:
    """必須キーのみを持つ最小設定を返す.

    Returns:
        PochiConfig.from_dict の入力に使う最小設定.
    """
    return {
        "model_name": "resnet18",
        "num_classes": 2,
        "device": "cpu",
        "epochs": 10,
        "batch_size": 8,
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "train_data_root": "data/train",
        "train_transform": object(),
        "val_transform": object(),
        "enable_layer_wise_lr": False,
    }


class TestPochiConfigFromDict:
    """PochiConfig.from_dict のテスト."""

    def test_missing_required_keys_raises_validation_error(self) -> None:
        """必須キー不足時に ValidationError になることを確認する."""
        with pytest.raises(ValidationError):
            PochiConfig.from_dict({})

    def test_optional_defaults_are_applied(self) -> None:
        """任意キー未指定時にデフォルト値が適用されることを確認する."""
        config = PochiConfig.from_dict(_build_minimum_config())

        assert config.pretrained is True
        assert config.cudnn_benchmark is False
        assert config.work_dir == "work_dirs"
        assert config.num_workers == 0
        assert config.train_pin_memory is True
        assert config.infer_pin_memory is True
        assert config.mean == [0.485, 0.456, 0.406]
        assert config.std == [0.229, 0.224, 0.225]
        assert config.early_stopping is None
        assert config.confusion_matrix_config is None
        assert config.optuna is None
        assert config.gradient_tracking_config.record_frequency == 1
        assert config.layer_wise_lr_config.layer_rates == {}

    def test_nested_configs(self) -> None:
        """ネスト設定が正しく反映されることを確認する."""
        payload = _build_minimum_config()
        payload.update(
            {
                "early_stopping": {
                    "enabled": True,
                    "patience": 3,
                    "min_delta": 0.01,
                    "monitor": "val_loss",
                },
                "confusion_matrix_config": {
                    "title": "CM",
                    "figsize": (12, 10),
                    "cmap": "Reds",
                },
                "gradient_tracking_config": {
                    "record_frequency": 2,
                    "exclude_patterns": ["layer4\\."],
                    "group_by_block": False,
                    "aggregation_method": "mean",
                },
                "layer_wise_lr_config": {
                    "layer_rates": {"fc": 0.01},
                    "graph_config": {"use_log_scale": False},
                },
                "optuna": {
                    "n_trials": 2,
                    "study_name": "nested_name",
                    "search_space": {"lr": {"type": "float"}},
                },
            }
        )

        config = PochiConfig.from_dict(payload)

        assert config.early_stopping is not None
        assert config.early_stopping.enabled is True
        assert config.early_stopping.patience == 3

        assert config.confusion_matrix_config is not None
        assert config.confusion_matrix_config.title == "CM"
        assert config.confusion_matrix_config.figsize == (12, 10)
        assert config.confusion_matrix_config.cmap == "Reds"

        assert config.gradient_tracking_config.record_frequency == 2
        assert config.gradient_tracking_config.exclude_patterns == ["layer4\\."]
        assert config.gradient_tracking_config.group_by_block is False
        assert config.gradient_tracking_config.aggregation_method == "mean"

        assert config.layer_wise_lr_config.layer_rates["fc"] == 0.01
        assert config.layer_wise_lr_config.graph_config.use_log_scale is False

        assert config.optuna is not None
        assert config.optuna.n_trials == 2
        assert config.optuna.study_name == "nested_name"
        assert "lr" in config.optuna.search_space

    def test_early_stopping_none_when_not_provided(self) -> None:
        """early_stopping キー未指定で None になることを確認する."""
        config = PochiConfig.from_dict(_build_minimum_config())
        assert config.early_stopping is None

    def test_early_stopping_constructed_when_provided(self) -> None:
        """early_stopping を dict で指定した場合に構築されることを確認する."""
        payload = _build_minimum_config()
        payload["early_stopping"] = {"enabled": True, "patience": 5}
        config = PochiConfig.from_dict(payload)

        assert config.early_stopping is not None
        assert config.early_stopping.patience == 5


class TestPochiConfigFieldValidation:
    """PochiConfig のフィールドバリデーションテスト.

    旧 validation/ のテスト観点を移植.
    """

    def test_unsupported_model_name_raises_error(self) -> None:
        """サポート外のモデル名で ValidationError になることを確認する."""
        payload = _build_minimum_config()
        payload["model_name"] = "vgg16"
        with pytest.raises(ValidationError):
            PochiConfig.from_dict(payload)

    def test_zero_num_classes_raises_error(self) -> None:
        """num_classes が 0 の場合に ValidationError になることを確認する."""
        payload = _build_minimum_config()
        payload["num_classes"] = 0
        with pytest.raises(ValidationError):
            PochiConfig.from_dict(payload)

    def test_bool_num_classes_raises_error(self) -> None:
        """num_classes に bool を指定すると ValidationError になることを確認する."""
        payload = _build_minimum_config()
        payload["num_classes"] = True
        with pytest.raises(
            ValidationError, match="bool は整数設定として使用できません"
        ):
            PochiConfig.from_dict(payload)

    def test_zero_epochs_raises_error(self) -> None:
        """epochs が 0 の場合に ValidationError になることを確認する."""
        payload = _build_minimum_config()
        payload["epochs"] = 0
        with pytest.raises(ValidationError):
            PochiConfig.from_dict(payload)

    def test_bool_epochs_raises_error(self) -> None:
        """epochs に bool を指定すると ValidationError になることを確認する."""
        payload = _build_minimum_config()
        payload["epochs"] = True
        with pytest.raises(
            ValidationError, match="bool は整数設定として使用できません"
        ):
            PochiConfig.from_dict(payload)

    def test_negative_batch_size_raises_error(self) -> None:
        """batch_size が負の場合に ValidationError になることを確認する."""
        payload = _build_minimum_config()
        payload["batch_size"] = -1
        with pytest.raises(ValidationError):
            PochiConfig.from_dict(payload)

    def test_bool_batch_size_raises_error(self) -> None:
        """batch_size に bool を指定すると ValidationError になることを確認する."""
        payload = _build_minimum_config()
        payload["batch_size"] = True
        with pytest.raises(
            ValidationError, match="bool は整数設定として使用できません"
        ):
            PochiConfig.from_dict(payload)

    def test_zero_learning_rate_raises_error(self) -> None:
        """learning_rate が 0 の場合に ValidationError になることを確認する."""
        payload = _build_minimum_config()
        payload["learning_rate"] = 0
        with pytest.raises(ValidationError):
            PochiConfig.from_dict(payload)

    def test_learning_rate_exceeds_upper_bound_raises_error(self) -> None:
        """learning_rate が 1.0 を超える場合に ValidationError になることを確認する."""
        payload = _build_minimum_config()
        payload["learning_rate"] = 1.5
        with pytest.raises(ValidationError):
            PochiConfig.from_dict(payload)

    def test_learning_rate_boundary_1_0_is_valid(self) -> None:
        """learning_rate = 1.0 が有効であることを確認する."""
        payload = _build_minimum_config()
        payload["learning_rate"] = 1.0
        config = PochiConfig.from_dict(payload)
        assert config.learning_rate == 1.0

    def test_unsupported_optimizer_raises_error(self) -> None:
        """サポート外のオプティマイザで ValidationError になることを確認する."""
        payload = _build_minimum_config()
        payload["optimizer"] = "RMSprop"
        with pytest.raises(ValidationError):
            PochiConfig.from_dict(payload)

    def test_invalid_device_raises_error(self) -> None:
        """不正なデバイス指定で ValidationError になることを確認する."""
        payload = _build_minimum_config()
        payload["device"] = "tpu"
        with pytest.raises(ValidationError):
            PochiConfig.from_dict(payload)

    def test_empty_train_data_root_raises_error(self) -> None:
        """train_data_root が空文字の場合に ValidationError になることを確認する."""
        payload = _build_minimum_config()
        payload["train_data_root"] = ""
        with pytest.raises(ValidationError):
            PochiConfig.from_dict(payload)

    def test_none_train_transform_raises_error(self) -> None:
        """train_transform が None の場合に ValidationError になることを確認する."""
        payload = _build_minimum_config()
        payload["train_transform"] = None
        with pytest.raises(ValidationError):
            PochiConfig.from_dict(payload)

    def test_none_val_transform_raises_error(self) -> None:
        """val_transform が None の場合に ValidationError になることを確認する."""
        payload = _build_minimum_config()
        payload["val_transform"] = None
        with pytest.raises(ValidationError):
            PochiConfig.from_dict(payload)


class TestPochiConfigSchedulerValidation:
    """scheduler と scheduler_params の整合性テスト.

    旧 SchedulerValidator のテスト観点を移植.
    """

    def test_scheduler_none_is_valid(self) -> None:
        """scheduler が None の場合に成功することを確認する."""
        payload = _build_minimum_config()
        payload["scheduler"] = None
        config = PochiConfig.from_dict(payload)
        assert config.scheduler is None

    def test_unsupported_scheduler_raises_error(self) -> None:
        """サポート外のスケジューラで ValidationError になることを確認する."""
        payload = _build_minimum_config()
        payload["scheduler"] = "UnsupportedLR"
        payload["scheduler_params"] = {}
        with pytest.raises(ValidationError):
            PochiConfig.from_dict(payload)

    def test_scheduler_without_params_raises_error(self) -> None:
        """scheduler_params なしでスケジューラ指定すると ValidationError になることを確認する."""
        payload = _build_minimum_config()
        payload["scheduler"] = "StepLR"
        with pytest.raises(ValidationError, match="scheduler_params は必須"):
            PochiConfig.from_dict(payload)

    def test_step_lr_missing_step_size_raises_error(self) -> None:
        """StepLR で step_size 未指定の場合に ValidationError になることを確認する."""
        payload = _build_minimum_config()
        payload["scheduler"] = "StepLR"
        payload["scheduler_params"] = {"gamma": 0.1}
        with pytest.raises(ValidationError, match="step_size"):
            PochiConfig.from_dict(payload)

    def test_step_lr_valid(self) -> None:
        """StepLR の正常な設定が成功することを確認する."""
        payload = _build_minimum_config()
        payload["scheduler"] = "StepLR"
        payload["scheduler_params"] = {"step_size": 30, "gamma": 0.1}
        config = PochiConfig.from_dict(payload)
        assert config.scheduler == "StepLR"

    def test_multi_step_lr_missing_milestones_raises_error(self) -> None:
        """MultiStepLR で milestones 未指定の場合に ValidationError になることを確認する."""
        payload = _build_minimum_config()
        payload["scheduler"] = "MultiStepLR"
        payload["scheduler_params"] = {"gamma": 0.1}
        with pytest.raises(ValidationError, match="milestones"):
            PochiConfig.from_dict(payload)

    def test_cosine_annealing_lr_missing_t_max_raises_error(self) -> None:
        """CosineAnnealingLR で T_max 未指定の場合に ValidationError になることを確認する."""
        payload = _build_minimum_config()
        payload["scheduler"] = "CosineAnnealingLR"
        payload["scheduler_params"] = {}
        with pytest.raises(ValidationError, match="T_max"):
            PochiConfig.from_dict(payload)

    def test_exponential_lr_missing_gamma_raises_error(self) -> None:
        """ExponentialLR で gamma 未指定の場合に ValidationError になることを確認する."""
        payload = _build_minimum_config()
        payload["scheduler"] = "ExponentialLR"
        payload["scheduler_params"] = {}
        with pytest.raises(ValidationError, match="gamma"):
            PochiConfig.from_dict(payload)

    def test_linear_lr_missing_total_iters_raises_error(self) -> None:
        """LinearLR で total_iters 未指定の場合に ValidationError になることを確認する."""
        payload = _build_minimum_config()
        payload["scheduler"] = "LinearLR"
        payload["scheduler_params"] = {"start_factor": 1.0}
        with pytest.raises(ValidationError, match="total_iters"):
            PochiConfig.from_dict(payload)


class TestPochiConfigClassWeightsValidation:
    """class_weights のバリデーションテスト.

    旧 ClassWeightsValidator のテスト観点を移植.
    """

    def test_class_weights_none_is_valid(self) -> None:
        """class_weights が None の場合に成功することを確認する."""
        payload = _build_minimum_config()
        payload["class_weights"] = None
        config = PochiConfig.from_dict(payload)
        assert config.class_weights is None

    def test_class_weights_length_mismatch_raises_error(self) -> None:
        """class_weights の要素数が num_classes と一致しない場合に ValidationError になることを確認する."""
        payload = _build_minimum_config()
        payload["num_classes"] = 4
        payload["class_weights"] = [1.0, 2.0]
        with pytest.raises(ValidationError, match="要素数"):
            PochiConfig.from_dict(payload)

    def test_class_weights_negative_value_raises_error(self) -> None:
        """class_weights に負の値がある場合に ValidationError になることを確認する."""
        payload = _build_minimum_config()
        payload["num_classes"] = 3
        payload["class_weights"] = [1.0, -2.0, 3.0]
        with pytest.raises(ValidationError, match="正の値"):
            PochiConfig.from_dict(payload)

    def test_class_weights_zero_value_raises_error(self) -> None:
        """class_weights に 0 がある場合に ValidationError になることを確認する."""
        payload = _build_minimum_config()
        payload["num_classes"] = 3
        payload["class_weights"] = [1.0, 0.0, 3.0]
        with pytest.raises(ValidationError, match="正の値"):
            PochiConfig.from_dict(payload)

    def test_valid_class_weights(self) -> None:
        """正常な class_weights が成功することを確認する."""
        payload = _build_minimum_config()
        payload["num_classes"] = 3
        payload["class_weights"] = [1.0, 2.5, 0.8]
        config = PochiConfig.from_dict(payload)
        assert config.class_weights == [1.0, 2.5, 0.8]


class TestPochiConfigLayerWiseLRValidation:
    """layer_wise_lr のバリデーションテスト.

    旧 LayerWiseLRValidator のテスト観点を移植.
    """

    def test_disabled_with_no_config_is_valid(self) -> None:
        """enable_layer_wise_lr が False の場合に成功することを確認する."""
        payload = _build_minimum_config()
        payload["enable_layer_wise_lr"] = False
        config = PochiConfig.from_dict(payload)
        assert config.enable_layer_wise_lr is False

    def test_enabled_with_empty_layer_rates_raises_error(self) -> None:
        """有効時に layer_rates が空の場合に ValidationError になることを確認する."""
        payload = _build_minimum_config()
        payload["enable_layer_wise_lr"] = True
        payload["layer_wise_lr_config"] = {"layer_rates": {}}
        with pytest.raises(ValidationError, match="layer_rates は空にできません"):
            PochiConfig.from_dict(payload)

    def test_enabled_with_no_config_raises_error(self) -> None:
        """有効時に layer_wise_lr_config がデフォルト(空)の場合に ValidationError になることを確認する."""
        payload = _build_minimum_config()
        payload["enable_layer_wise_lr"] = True
        with pytest.raises(ValidationError, match="layer_rates は空にできません"):
            PochiConfig.from_dict(payload)

    def test_enabled_with_negative_rate_raises_error(self) -> None:
        """有効時に学習率が負の場合に ValidationError になることを確認する."""
        payload = _build_minimum_config()
        payload["enable_layer_wise_lr"] = True
        payload["layer_wise_lr_config"] = {"layer_rates": {"conv1": -0.001}}
        with pytest.raises(ValidationError, match="正の値"):
            PochiConfig.from_dict(payload)

    def test_enabled_with_zero_rate_raises_error(self) -> None:
        """有効時に学習率が 0 の場合に ValidationError になることを確認する."""
        payload = _build_minimum_config()
        payload["enable_layer_wise_lr"] = True
        payload["layer_wise_lr_config"] = {"layer_rates": {"conv1": 0.0}}
        with pytest.raises(ValidationError, match="正の値"):
            PochiConfig.from_dict(payload)

    def test_enabled_with_valid_config(self) -> None:
        """有効時に正常な設定が成功することを確認する."""
        payload = _build_minimum_config()
        payload["enable_layer_wise_lr"] = True
        payload["layer_wise_lr_config"] = {"layer_rates": {"layer4": 0.001, "fc": 0.01}}
        config = PochiConfig.from_dict(payload)
        assert config.enable_layer_wise_lr is True
        assert config.layer_wise_lr_config.layer_rates["fc"] == 0.01


class TestPochiConfigRealConfigFile:
    """実サンプル設定ファイルの読み込みテスト."""

    def test_sample_config_has_nested_optuna(self) -> None:
        """sample config の optuna ネスト設定が読み込めることを確認する."""
        config_dict = ConfigLoader.load_config("configs/pochi_train_config.py")
        config = PochiConfig.from_dict(config_dict)

        assert config.optuna is not None
        assert "learning_rate" in config.optuna.search_space
