"""ベンチマーク結果の出力処理."""

import logging
from pathlib import Path
from typing import Optional

from pochitrain.inference.benchmark.env_name import resolve_env_name
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


def resolve_benchmark_env_name(
    *,
    use_gpu: bool,
    cli_env_name: Optional[str],
    config_env_name: Optional[str],
) -> str:
    """CLI 引数と config から環境名を組み立てて resolve_env_name() に委譲する.

    Args:
        use_gpu: GPU を利用しているかどうか.
        cli_env_name: CLI の --benchmark-env-name で指定された値.
        config_env_name: config の benchmark_env_name の値.

    Returns:
        環境識別文字列.
    """
    configured = cli_env_name or config_env_name
    return resolve_env_name(
        use_gpu=use_gpu,
        configured_env_name=str(configured) if configured is not None else None,
    )


def export_benchmark_json(
    output_dir: Path,
    benchmark_result: BenchmarkResult,
    logger: logging.Logger,
) -> None:
    """ベンチマーク結果 JSON の書き出しとログ出力を行う.

    Args:
        output_dir: 出力ディレクトリ.
        benchmark_result: 保存するベンチ結果.
        logger: ロガー.
    """
    try:
        json_path = write_benchmark_result_json(
            output_dir=output_dir,
            benchmark_result=benchmark_result,
        )
        logger.info(f"ベンチマークJSONを出力しました: {json_path.name}")
    except Exception as exc:  # pragma: no cover
        logger.warning(f"ベンチマークJSONの保存に失敗しました, error: {exc}")
