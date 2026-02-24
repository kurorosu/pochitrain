"""推論ベンチマークの実行・集計 CLI."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

from pochitrain.benchmark.aggregator import aggregate_results
from pochitrain.benchmark.loader import load_suite_config
from pochitrain.benchmark.runner import run_suite
from pochitrain.benchmark.utils import configure_logger

LOGGER = logging.getLogger("pochitrain.benchmark")
BENCHMARK_OUTPUT_ROOT = Path("benchmark_runs")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """CLI 引数を解析する.

    Args:
        argv: 解析対象引数. 省略時は `sys.argv`.

    Returns:
        解析済み引数.
    """
    parser = argparse.ArgumentParser(
        description="推論ベンチマークの実行・集計スクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--suite", default=None, help="suites.yaml のスイート名")
    parser.add_argument(
        "--suites-file",
        default="configs/bench_suites.yaml",
        help="スイート定義ファイルパス",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="実行はせず既存結果の集計のみ行う",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="aggregate-only 時の入力ディレクトリ",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="ケース失敗時に即座に終了する",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="デバッグログを有効化する",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """エントリポイント.

    Args:
        argv: 解析対象引数. 省略時は `sys.argv`.

    Returns:
        終了コード.
    """
    args = parse_args(argv)
    configure_logger(args.debug)

    if args.aggregate_only:
        if not args.input_dir:
            LOGGER.error("--aggregate-only では --input-dir が必要です.")
            return 1
        try:
            csv_path, json_path = aggregate_results(Path(args.input_dir))
            LOGGER.info("summary csv : %s", csv_path)
            LOGGER.info("summary json: %s", json_path)
            return 0
        except Exception as exc:
            LOGGER.error("集計に失敗しました: %s", exc)
            return 1

    if not args.suite:
        LOGGER.error("実行モードでは --suite が必要です.")
        return 1

    try:
        suite = load_suite_config(Path(args.suites_file), args.suite)
    except Exception as exc:
        LOGGER.error("スイート読み込みに失敗しました: %s", exc)
        return 1

    try:
        run_dir = run_suite(
            suite=suite,
            output_root=BENCHMARK_OUTPUT_ROOT,
            fail_fast=args.fail_fast,
        )
    except Exception as exc:
        LOGGER.error("ベンチ実行に失敗しました: %s", exc)
        return 1

    try:
        csv_path, json_path = aggregate_results(run_dir)
        LOGGER.info("run dir: %s", run_dir)
        LOGGER.info("summary csv : %s", csv_path)
        LOGGER.info("summary json: %s", json_path)
        return 0
    except Exception as exc:
        LOGGER.error("集計に失敗しました: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
