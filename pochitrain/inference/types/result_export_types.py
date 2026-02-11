"""推論結果出力で共有する型定義."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ResultExportRequest:
    """ResultExportService へ渡す入力パラメータ."""

    output_dir: Path
    model_path: Path
    data_path: Path
    image_paths: List[str]
    predictions: List[int]
    true_labels: List[int]
    confidences: List[float]
    class_names: List[str]
    num_samples: int
    correct: int
    avg_time_per_image: float
    total_samples: int
    warmup_samples: int
    avg_total_time_per_image: Optional[float] = None
    input_size: Optional[Tuple[int, int, int]] = None
    results_filename: str = "inference_results.csv"
    summary_filename: str = "inference_summary.txt"
    confusion_matrix_filename: str = "confusion_matrix.png"
    classification_report_filename: str = "classification_report.csv"
    extra_info: Optional[Dict[str, Any]] = None
    cm_config: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class ResultExportResult:
    """ResultExportService から返却される出力値."""

    results_csv_path: Path
    summary_path: Path
    confusion_matrix_path: Optional[Path]
    classification_report_path: Optional[Path]
    accuracy: float
