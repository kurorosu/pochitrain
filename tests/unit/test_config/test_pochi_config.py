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
    """PochiConfig のフィールドバリデーションテスト."""

    @pytest.mark.parametrize(
        "field, value, match",
        [
            ("model_name", "vgg16", None),
            ("num_classes", 0, None),
            ("num_classes", True, "bool は整数設定として使用できません"),
            ("epochs", 0, None),
            ("epochs", True, "bool は整数設定として使用できません"),
            ("batch_size", -1, None),
            ("batch_size", True, "bool は整数設定として使用できません"),
            ("learning_rate", 0, None),
            ("learning_rate", 1.5, None),
            ("optimizer", "RMSprop", None),
            ("device", "tpu", None),
            ("train_data_root", "", None),
            ("train_transform", None, None),
            ("val_transform", None, None),
        ],
    )
    def test_invalid_field_values_raise_error(self, field, value, match) -> None:
        """不正なフィールド値で ValidationError になることを確認する."""
        payload = _build_minimum_config()
        payload[field] = value
        with pytest.raises(ValidationError, match=match):
            PochiConfig.from_dict(payload)

    def test_learning_rate_boundary_1_0_is_valid(self) -> None:
        """learning_rate = 1.0 が有効であることを確認する."""
        payload = _build_minimum_config()
        payload["learning_rate"] = 1.0
        config = PochiConfig.from_dict(payload)
        assert config.learning_rate == 1.0


class TestPochiConfigSchedulerValidation:
    """scheduler と scheduler_params の整合性テスト."""

    def test_scheduler_none_is_valid(self) -> None:
        """scheduler が None の場合に成功することを確認する."""
        payload = _build_minimum_config()
        payload["scheduler"] = None
        config = PochiConfig.from_dict(payload)
        assert config.scheduler is None

    @pytest.mark.parametrize(
        "scheduler, params, match",
        [
            ("UnsupportedLR", {}, None),
            ("StepLR", None, "scheduler_params は必須"),
            ("StepLR", {"gamma": 0.1}, "step_size"),
            ("MultiStepLR", {"gamma": 0.1}, "milestones"),
            ("CosineAnnealingLR", {}, "T_max"),
            ("ExponentialLR", {}, "gamma"),
            ("LinearLR", {"start_factor": 1.0}, "total_iters"),
        ],
    )
    def test_invalid_scheduler_configs_raise_error(
        self, scheduler, params, match
    ) -> None:
        """不正なスケジューラ設定で ValidationError になることを確認する."""
        payload = _build_minimum_config()
        payload["scheduler"] = scheduler
        if params is not None:
            payload["scheduler_params"] = params
        else:
            payload.pop("scheduler_params", None)

        with pytest.raises(ValidationError, match=match):
            PochiConfig.from_dict(payload)

    def test_step_lr_valid(self) -> None:
        """StepLR の正常な設定が成功することを確認する."""
        payload = _build_minimum_config()
        payload["scheduler"] = "StepLR"
        payload["scheduler_params"] = {"step_size": 30, "gamma": 0.1}
        config = PochiConfig.from_dict(payload)
        assert config.scheduler == "StepLR"


class TestPochiConfigClassWeightsValidation:
    """class_weights のバリデーションテスト."""

    def test_class_weights_none_is_valid(self) -> None:
        """class_weights が None の場合に成功することを確認する."""
        payload = _build_minimum_config()
        payload["class_weights"] = None
        config = PochiConfig.from_dict(payload)
        assert config.class_weights is None

    @pytest.mark.parametrize(
        "num_classes, weights, match",
        [
            (4, [1.0, 2.0], "要素数"),
            (3, [1.0, -2.0, 3.0], "正の値"),
            (3, [1.0, 0.0, 3.0], "正の値"),
        ],
    )
    def test_invalid_class_weights_raise_error(
        self, num_classes, weights, match
    ) -> None:
        """不正な class_weights で ValidationError になることを確認する."""
        payload = _build_minimum_config()
        payload["num_classes"] = num_classes
        payload["class_weights"] = weights
        with pytest.raises(ValidationError, match=match):
            PochiConfig.from_dict(payload)

    def test_valid_class_weights(self) -> None:
        """正常な class_weights が成功することを確認する."""
        payload = _build_minimum_config()
        payload["num_classes"] = 3
        payload["class_weights"] = [1.0, 2.5, 0.8]
        config = PochiConfig.from_dict(payload)
        assert config.class_weights == [1.0, 2.5, 0.8]


class TestPochiConfigLayerWiseLRValidation:
    """layer_wise_lr のバリデーションテスト."""

    def test_disabled_with_no_config_is_valid(self) -> None:
        """enable_layer_wise_lr が False の場合に成功することを確認する."""
        payload = _build_minimum_config()
        payload["enable_layer_wise_lr"] = False
        config = PochiConfig.from_dict(payload)
        assert config.enable_layer_wise_lr is False

    @pytest.mark.parametrize(
        "config_payload, match",
        [
            ({"layer_rates": {}}, "layer_rates は空にできません"),
            (None, "layer_rates は空にできません"),
            ({"layer_rates": {"conv1": -0.001}}, "正の値"),
            ({"layer_rates": {"conv1": 0.0}}, "正の値"),
        ],
    )
    def test_invalid_layer_wise_lr_configs_raise_error(
        self, config_payload, match
    ) -> None:
        """不正な layer_wise_lr 設定で ValidationError になることを確認する."""
        payload = _build_minimum_config()
        payload["enable_layer_wise_lr"] = True
        if config_payload is not None:
            payload["layer_wise_lr_config"] = config_payload

        with pytest.raises(ValidationError, match=match):
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
