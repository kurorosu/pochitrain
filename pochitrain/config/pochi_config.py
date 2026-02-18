"""pochitrain.config.pochi_config: 型付き設定のメイン dataclass."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from .sub_configs import (
    ConfusionMatrixConfig,
    EarlyStoppingConfig,
    GradientTrackingConfig,
    LayerWiseLRConfig,
    OptunaConfig,
)

_OPTUNA_KEYS = {
    "search_space",
    "n_trials",
    "n_jobs",
    "optuna_epochs",
    "study_name",
    "direction",
    "sampler",
    "pruner",
    "storage",
}


@dataclass
class PochiConfig:
    """学習と推論の型付き設定."""

    # Required
    model_name: str
    num_classes: int
    device: str
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str
    train_data_root: str
    train_transform: Any
    val_transform: Any
    enable_layer_wise_lr: bool

    # Optional with defaults
    val_data_root: Optional[str] = None
    pretrained: bool = True
    cudnn_benchmark: bool = False
    class_weights: Optional[List[float]] = None
    scheduler: Optional[str] = None
    scheduler_params: Optional[Dict[str, Any]] = None
    work_dir: str = "work_dirs"
    num_workers: int = 0
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    enable_metrics_export: bool = True
    enable_gradient_tracking: bool = False
    gradient_tracking_config: GradientTrackingConfig = field(
        default_factory=GradientTrackingConfig
    )
    confusion_matrix_config: Optional[ConfusionMatrixConfig] = None
    layer_wise_lr_config: LayerWiseLRConfig = field(default_factory=LayerWiseLRConfig)
    early_stopping: Optional[EarlyStoppingConfig] = None
    optuna: Optional[OptunaConfig] = None

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "PochiConfig":
        """Dict から PochiConfig を生成."""
        required_keys = [
            "model_name",
            "num_classes",
            "device",
            "epochs",
            "batch_size",
            "learning_rate",
            "optimizer",
            "train_data_root",
            "train_transform",
            "val_transform",
            "enable_layer_wise_lr",
        ]
        missing = [key for key in required_keys if key not in config]
        if missing:
            missing_keys = ", ".join(missing)
            raise ValueError(f"Missing required config keys: {missing_keys}")

        early_stopping = None
        if config.get("early_stopping") is not None:
            early_stopping = EarlyStoppingConfig.model_validate(
                config["early_stopping"]
            )

        confusion_matrix_config = None
        if config.get("confusion_matrix_config") is not None:
            confusion_matrix_config = ConfusionMatrixConfig.model_validate(
                config["confusion_matrix_config"]
            )

        gradient_tracking_config = GradientTrackingConfig.model_validate(
            config.get("gradient_tracking_config") or {}
        )
        layer_wise_lr_config = LayerWiseLRConfig.model_validate(
            config.get("layer_wise_lr_config") or {}
        )

        optuna_value = config.get("optuna")
        optuna_data = optuna_value if isinstance(optuna_value, dict) else {}
        has_optuna_key = any(key in config for key in _OPTUNA_KEYS)
        has_optuna_dict = bool(optuna_data)
        optuna = None
        if has_optuna_key or has_optuna_dict:
            merged_optuna = {
                **optuna_data,
                **{key: config.get(key) for key in _OPTUNA_KEYS if key in config},
            }
            optuna = OptunaConfig.model_validate(merged_optuna)

        return cls(
            model_name=config["model_name"],
            num_classes=config["num_classes"],
            device=config["device"],
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            optimizer=config["optimizer"],
            train_data_root=config["train_data_root"],
            val_data_root=config.get("val_data_root"),
            train_transform=config["train_transform"],
            val_transform=config["val_transform"],
            enable_layer_wise_lr=config["enable_layer_wise_lr"],
            pretrained=config.get("pretrained", True),
            cudnn_benchmark=config.get("cudnn_benchmark", False),
            class_weights=config.get("class_weights"),
            scheduler=config.get("scheduler"),
            scheduler_params=config.get("scheduler_params"),
            work_dir=config.get("work_dir", "work_dirs"),
            num_workers=config.get("num_workers", 0),
            mean=config.get("mean", [0.485, 0.456, 0.406]),
            std=config.get("std", [0.229, 0.224, 0.225]),
            enable_metrics_export=config.get("enable_metrics_export", True),
            enable_gradient_tracking=config.get("enable_gradient_tracking", False),
            gradient_tracking_config=gradient_tracking_config,
            confusion_matrix_config=confusion_matrix_config,
            layer_wise_lr_config=layer_wise_lr_config,
            early_stopping=early_stopping,
            optuna=optuna,
        )

    def to_dict(self) -> Dict[str, Any]:
        """既存モジュール互換のため Dict に変換."""
        result: Dict[str, Any] = {}
        for field_info in dataclasses.fields(self):
            value = getattr(self, field_info.name)
            if field_info.name == "optuna":
                if value is not None:
                    result.update(value.model_dump())
                continue
            if isinstance(value, BaseModel):
                result[field_info.name] = value.model_dump()
            else:
                result[field_info.name] = value
        return result
