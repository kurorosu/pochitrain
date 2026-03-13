"""推論結果出力で共有する型定義."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class ResultExportRequest:
    """ResultExportService へ渡す入力パラメータ."""

    output_dir: Path
    model_path: Path
    data_path: Path
    image_paths: list[str]
    predictions: list[int]
    true_labels: list[int]
    confidences: list[float]
    class_names: list[str]
    num_samples: int
    correct: int
    avg_time_per_image: float
    total_samples: int
    warmup_samples: int
    model_info: Optional[dict[str, Any]] = None
    avg_total_time_per_image: Optional[float] = None
    input_size: Optional[tuple[int, int, int]] = None
    results_filename: str = "inference_results.csv"
    summary_filename: str = "inference_summary.txt"
    confusion_matrix_filename: str = "confusion_matrix.png"
    classification_report_filename: str = "classification_report.csv"
    model_info_filename: str = "model_info.json"
    extra_info: Optional[dict[str, Any]] = None
    cm_config: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class ResultExportResult:
    """ResultExportService から返却される出力値."""

    results_csv_path: Path
    summary_path: Path
    confusion_matrix_path: Optional[Path]
    classification_report_path: Optional[Path]
    model_info_path: Optional[Path]
    accuracy: float
