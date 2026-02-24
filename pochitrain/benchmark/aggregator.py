"""ベンチマーク結果の集計処理."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, DefaultDict, Dict, List, Tuple

from pochitrain.benchmark.utils import now_jst_timestamp, to_float, write_json


def _collect_benchmark_paths(input_dir: Path) -> List[Path]:
    """集計対象の benchmark_result.json 一覧を返す.

    Args:
        input_dir: 集計対象ディレクトリ.

    Returns:
        benchmark_result.json のパス一覧.
    """
    search_root = input_dir / "raw" if (input_dir / "raw").exists() else input_dir
    return sorted(search_root.rglob("benchmark_result.json"))


def _extract_case_name(path: Path) -> str:
    """結果ファイルパスからケース名を抽出する.

    Args:
        path: benchmark_result.json パス.

    Returns:
        ケース名. 抽出できない場合は空文字.
    """
    case_dir_name = path.parent.parent.name
    if case_dir_name.startswith("case_"):
        parts = case_dir_name.split("_", 2)
        if len(parts) == 3:
            return parts[2]
    return ""


def aggregate_results(input_dir: Path) -> Tuple[Path, Path]:
    """ベンチ結果を集計する.

    Args:
        input_dir: 集計対象ディレクトリ.

    Returns:
        `benchmark_summary.csv` と `benchmark_summary.json` のパス.
    """
    benchmark_paths = _collect_benchmark_paths(input_dir)
    if len(benchmark_paths) == 0:
        raise ValueError(f"benchmark_result.json が見つかりません: {input_dir}")

    groups: DefaultDict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    invalid_files: List[Dict[str, str]] = []

    for path in benchmark_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            invalid_files.append({"path": str(path), "error": str(exc)})
            continue

        metrics = payload.get("metrics", {})
        case_name = _extract_case_name(path)
        key = (
            case_name,
            payload.get("env_name"),
            payload.get("runtime"),
            payload.get("precision"),
            payload.get("model_name"),
            payload.get("pipeline"),
            payload.get("device"),
        )
        groups[key].append({"metrics": metrics, "path": path})

    rows: List[Dict[str, Any]] = []
    for key in sorted(groups.keys(), key=lambda item: tuple(str(v) for v in item)):
        entries = groups[key]
        inference_values = [
            x
            for x in (
                to_float(entry["metrics"].get("avg_inference_ms")) for entry in entries
            )
            if x is not None
        ]
        e2e_values = [
            x
            for x in (to_float(entry["metrics"].get("avg_e2e_ms")) for entry in entries)
            if x is not None
        ]
        acc_values = [
            x
            for x in (
                to_float(entry["metrics"].get("accuracy_percent")) for entry in entries
            )
            if x is not None
        ]

        rows.append(
            {
                "case_name": key[0],
                "env_name": key[1],
                "runtime": key[2],
                "precision": key[3],
                "model_name": key[4],
                "pipeline": key[5],
                "device": key[6],
                "runs": len(entries),
                "avg_inference_ms_mean": (
                    mean(inference_values) if inference_values else None
                ),
                "avg_inference_ms_stdev": (
                    pstdev(inference_values) if len(inference_values) >= 2 else 0.0
                ),
                "avg_e2e_ms_mean": mean(e2e_values) if e2e_values else None,
                "avg_e2e_ms_stdev": pstdev(e2e_values) if len(e2e_values) >= 2 else 0.0,
                "accuracy_percent_mean": mean(acc_values) if acc_values else None,
                "source_files": [str(entry["path"]) for entry in entries],
            }
        )

    summary_dir = input_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_csv_path = summary_dir / "benchmark_summary.csv"
    summary_json_path = summary_dir / "benchmark_summary.json"

    csv_headers = [
        "case_name",
        "env_name",
        "runtime",
        "precision",
        "model_name",
        "pipeline",
        "device",
        "runs",
        "avg_inference_ms_mean",
        "avg_inference_ms_stdev",
        "avg_e2e_ms_mean",
        "avg_e2e_ms_stdev",
        "accuracy_percent_mean",
    ]
    with open(summary_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({h: row.get(h) for h in csv_headers})

    write_json(
        summary_json_path,
        {
            "generated_at_jst": now_jst_timestamp(),
            "source_dir": str(input_dir),
            "total_benchmark_files": len(benchmark_paths),
            "total_groups": len(rows),
            "invalid_files": invalid_files,
            "groups": rows,
        },
    )
    return summary_csv_path, summary_json_path
