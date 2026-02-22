"""ベンチマーク結果の出力処理."""

import json
from pathlib import Path

from pochitrain.inference.types.benchmark_types import (
    BENCHMARK_RESULT_FILENAME,
    BenchmarkResult,
)


def write_benchmark_result_json(
    output_dir: Path,
    benchmark_result: BenchmarkResult,
    filename: str = BENCHMARK_RESULT_FILENAME,
) -> Path:
    """ベンチマーク結果JSONを保存する.

    Args:
        output_dir: 出力ディレクトリ.
        benchmark_result: 保存するベンチ結果.
        filename: 出力ファイル名.

    Returns:
        保存したJSONファイルのパス.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_result.to_dict(), f, ensure_ascii=False, indent=2)
    return output_path
