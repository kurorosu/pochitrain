"""ベンチマーク実行設定の型定義."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class CaseConfig:
    """ベンチマークケースの設定."""

    name: str
    runtime: str
    model_path: Path
    pipeline: str
    repeats: int
    benchmark_env_name: Optional[str]


@dataclass(frozen=True)
class SuiteConfig:
    """ベンチマークスイートの設定."""

    name: str
    description: str
    cases: List[CaseConfig]
