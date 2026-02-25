"""ベンチマーク結果の出力処理."""

from pathlib import Path

from pochitrain.inference.types.benchmark_types import (
    BENCHMARK_RESULT_FILENAME,
    BenchmarkResult,
)
from pochitrain.utils.json_utils import write_json_file


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
    output_path = output_dir / filename
    return write_json_file(output_path, benchmark_result.to_dict())
