"""pochitrain.config.sub_configs: ネスト設定用 Pydantic モデル定義."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field


class EarlyStoppingConfig(BaseModel):
    """早期終了設定."""

    enabled: bool = False
    patience: int = Field(default=10, gt=0)
    min_delta: float = Field(default=0.0, ge=0.0)
    monitor: Literal["val_accuracy", "val_loss"] = "val_accuracy"


class LayerWiseLRGraphConfig(BaseModel):
    """層別学習率グラフ設定."""

    use_log_scale: bool = True


class LayerWiseLRConfig(BaseModel):
    """層別学習率設定."""

    layer_rates: Dict[str, float] = Field(default_factory=dict)
    graph_config: LayerWiseLRGraphConfig = Field(default_factory=LayerWiseLRGraphConfig)


class GradientTrackingConfig(BaseModel):
    """勾配トラッキング設定."""

    record_frequency: int = Field(default=1, gt=0)
    exclude_patterns: List[str] = Field(default_factory=lambda: ["fc\\.", "\\.bias"])
    group_by_block: bool = True
    aggregation_method: str = "median"


class ConfusionMatrixConfig(BaseModel):
    """混同行列可視化設定."""

    title: str = "Confusion Matrix"
    xlabel: str = "Predicted Label"
    ylabel: str = "True Label"
    fontsize: int = Field(default=14, gt=0)
    title_fontsize: int = Field(default=16, gt=0)
    label_fontsize: int = Field(default=12, gt=0)
    figsize: Tuple[int, int] = (8, 6)
    cmap: str = "Blues"


class OptunaConfig(BaseModel):
    """Optuna 最適化設定."""

    search_space: Dict[str, Any] = Field(default_factory=dict)
    n_trials: int = Field(default=20, gt=0)
    n_jobs: int = Field(default=1, gt=0)
    optuna_epochs: int = Field(default=10, gt=0)
    study_name: str = "pochitrain_optimization"
    direction: str = "maximize"
    sampler: str = "TPESampler"
    pruner: Optional[str] = "MedianPruner"
    storage: Optional[str] = None
