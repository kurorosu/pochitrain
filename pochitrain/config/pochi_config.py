"""pochitrain.config.pochi_config: 型付き設定の Pydantic モデル."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .sub_configs import (
    ConfusionMatrixConfig,
    EarlyStoppingConfig,
    GradientTrackingConfig,
    LayerWiseLRConfig,
    OptunaConfig,
)

_SCHEDULER_REQUIRED_PARAMS: Dict[str, List[str]] = {
    "StepLR": ["step_size"],
    "MultiStepLR": ["milestones"],
    "CosineAnnealingLR": ["T_max"],
    "ExponentialLR": ["gamma"],
    "LinearLR": ["total_iters"],
}


class PochiConfig(BaseModel):
    """学習と推論の型付き設定."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Required
    model_name: Literal["resnet18", "resnet34", "resnet50"]
    num_classes: int = Field(gt=0)
    device: Literal["cuda", "cpu"]
    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0, le=1.0)
    optimizer: Literal["SGD", "Adam", "AdamW"]
    train_data_root: str
    train_transform: Any
    val_transform: Any
    enable_layer_wise_lr: bool

    val_data_root: Optional[str] = None
    pretrained: bool = True
    cudnn_benchmark: bool = False
    class_weights: Optional[List[float]] = None
    scheduler: Optional[
        Literal[
            "StepLR",
            "MultiStepLR",
            "CosineAnnealingLR",
            "ExponentialLR",
            "LinearLR",
        ]
    ] = None
    scheduler_params: Optional[Dict[str, Any]] = None
    work_dir: str = "work_dirs"
    num_workers: int = 0
    mean: List[float] = Field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = Field(default_factory=lambda: [0.229, 0.224, 0.225])
    enable_metrics_export: bool = True
    enable_gradient_tracking: bool = False
    gradient_tracking_config: GradientTrackingConfig = Field(
        default_factory=GradientTrackingConfig
    )
    confusion_matrix_config: Optional[ConfusionMatrixConfig] = None
    layer_wise_lr_config: LayerWiseLRConfig = Field(default_factory=LayerWiseLRConfig)
    early_stopping: Optional[EarlyStoppingConfig] = None
    optuna: Optional[OptunaConfig] = None

    @field_validator("train_data_root", "val_data_root", mode="before")
    @classmethod
    def path_must_not_be_empty(cls, v: Optional[str]) -> Optional[str]:
        """パスが空文字でないことを検証する."""
        if isinstance(v, str) and v.strip() == "":
            raise ValueError("パスは空文字を許可しません")
        return v

    @field_validator("train_transform", "val_transform")
    @classmethod
    def transform_must_exist(cls, v: Any) -> Any:
        """Transform が None でないことを検証する."""
        if v is None:
            raise ValueError("transform は必須です")
        return v

    @model_validator(mode="after")
    def validate_scheduler_params(self) -> "PochiConfig":
        """Scheduler と scheduler_params の整合性を検証する."""
        if self.scheduler is None:
            return self

        if self.scheduler_params is None:
            raise ValueError(
                f"scheduler '{self.scheduler}' を指定する場合、"
                "scheduler_params は必須です"
            )

        required = _SCHEDULER_REQUIRED_PARAMS.get(self.scheduler, [])
        for param in required:
            if param not in self.scheduler_params:
                raise ValueError(
                    f"scheduler '{self.scheduler}' には "
                    f"'{param}' パラメータが必須です"
                )

        return self

    @model_validator(mode="after")
    def validate_class_weights(self) -> "PochiConfig":
        """class_weights と num_classes の整合性を検証する."""
        if self.class_weights is None:
            return self

        if len(self.class_weights) != self.num_classes:
            raise ValueError(
                f"class_weights の要素数 ({len(self.class_weights)}) が "
                f"num_classes ({self.num_classes}) と一致しません"
            )

        for i, w in enumerate(self.class_weights):
            if w <= 0:
                raise ValueError(
                    f"class_weights[{i}] は正の値である必要があります (値: {w})"
                )

        return self

    @model_validator(mode="after")
    def validate_layer_wise_lr(self) -> "PochiConfig":
        """enable_layer_wise_lr と layer_wise_lr_config の整合性を検証する."""
        if not self.enable_layer_wise_lr:
            return self

        if not self.layer_wise_lr_config.layer_rates:
            raise ValueError(
                "enable_layer_wise_lr が True の場合、"
                "layer_wise_lr_config.layer_rates は空にできません"
            )

        for name, rate in self.layer_wise_lr_config.layer_rates.items():
            if rate <= 0:
                raise ValueError(
                    f"layer_rates['{name}'] は正の値である必要があります "
                    f"(値: {rate})"
                )

        return self

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "PochiConfig":
        """Dict から PochiConfig を生成. Pydantic バリデーションが自動実行される."""
        payload = dict(config)

        # サブ config の組み立て
        if payload.get("early_stopping") is not None:
            payload["early_stopping"] = EarlyStoppingConfig.model_validate(
                payload["early_stopping"]
            )

        if payload.get("confusion_matrix_config") is not None:
            payload["confusion_matrix_config"] = ConfusionMatrixConfig.model_validate(
                payload["confusion_matrix_config"]
            )

        payload["gradient_tracking_config"] = GradientTrackingConfig.model_validate(
            payload.get("gradient_tracking_config") or {}
        )
        payload["layer_wise_lr_config"] = LayerWiseLRConfig.model_validate(
            payload.get("layer_wise_lr_config") or {}
        )

        if payload.get("optuna") is not None:
            payload["optuna"] = OptunaConfig.model_validate(payload["optuna"])

        result: PochiConfig = cls.model_validate(payload)
        return result
