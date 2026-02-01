"""pochitrain.config.sub_configs: ネスト設定用 dataclass 定義."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class EarlyStoppingConfig:
    """早期終了設定."""

    enabled: bool = False
    patience: int = 10
    min_delta: float = 0.0
    monitor: str = "val_accuracy"

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "EarlyStoppingConfig":
        """Dict から設定を作成."""
        payload = data or {}
        return cls(
            enabled=payload.get("enabled", False),
            patience=payload.get("patience", 10),
            min_delta=payload.get("min_delta", 0.0),
            monitor=payload.get("monitor", "val_accuracy"),
        )


@dataclass
class LayerWiseLRGraphConfig:
    """層別学習率グラフ設定."""

    use_log_scale: bool = True

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "LayerWiseLRGraphConfig":
        """Dict から設定を作成."""
        payload = data or {}
        return cls(use_log_scale=payload.get("use_log_scale", True))


@dataclass
class LayerWiseLRConfig:
    """層別学習率設定."""

    layer_rates: Dict[str, float] = field(default_factory=dict)
    graph_config: LayerWiseLRGraphConfig = field(default_factory=LayerWiseLRGraphConfig)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "LayerWiseLRConfig":
        """Dict から設定を作成."""
        payload = data or {}
        graph_config = LayerWiseLRGraphConfig.from_dict(payload.get("graph_config"))
        return cls(
            layer_rates=payload.get("layer_rates", {}),
            graph_config=graph_config,
        )


@dataclass
class GradientTrackingConfig:
    """勾配トラッキング設定."""

    record_frequency: int = 1
    exclude_patterns: List[str] = field(default_factory=lambda: ["fc\\.", "\\.bias"])
    group_by_block: bool = True
    aggregation_method: str = "median"

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "GradientTrackingConfig":
        """Dict から設定を作成."""
        payload = data or {}
        return cls(
            record_frequency=payload.get("record_frequency", 1),
            exclude_patterns=payload.get("exclude_patterns", ["fc\\.", "\\.bias"]),
            group_by_block=payload.get("group_by_block", True),
            aggregation_method=payload.get("aggregation_method", "median"),
        )


@dataclass
class ConfusionMatrixConfig:
    """混同行列可視化設定."""

    title: str = "Confusion Matrix"
    xlabel: str = "Predicted Label"
    ylabel: str = "True Label"
    fontsize: int = 14
    title_fontsize: int = 16
    label_fontsize: int = 12
    figsize: Tuple[int, int] = (8, 6)
    cmap: str = "Blues"

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ConfusionMatrixConfig":
        """Dict から設定を作成."""
        payload = data or {}
        return cls(
            title=payload.get("title", "Confusion Matrix"),
            xlabel=payload.get("xlabel", "Predicted Label"),
            ylabel=payload.get("ylabel", "True Label"),
            fontsize=payload.get("fontsize", 14),
            title_fontsize=payload.get("title_fontsize", 16),
            label_fontsize=payload.get("label_fontsize", 12),
            figsize=payload.get("figsize", (8, 6)),
            cmap=payload.get("cmap", "Blues"),
        )


@dataclass
class OptunaConfig:
    """Optuna 最適化設定."""

    search_space: Dict[str, Any] = field(default_factory=dict)
    n_trials: int = 20
    n_jobs: int = 1
    optuna_epochs: int = 10
    study_name: str = "pochitrain_optimization"
    direction: str = "maximize"
    sampler: str = "TPESampler"
    pruner: Optional[str] = "MedianPruner"
    storage: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "OptunaConfig":
        """Dict から設定を作成."""
        payload = data or {}
        return cls(
            search_space=payload.get("search_space", {}),
            n_trials=payload.get("n_trials", 20),
            n_jobs=payload.get("n_jobs", 1),
            optuna_epochs=payload.get("optuna_epochs", 10),
            study_name=payload.get("study_name", "pochitrain_optimization"),
            direction=payload.get("direction", "maximize"),
            sampler=payload.get("sampler", "TPESampler"),
            pruner=payload.get("pruner", "MedianPruner"),
            storage=payload.get("storage", None),
        )
