"""ベンチマークケース実行処理."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

from pochitrain.benchmark.models import CaseConfig, SuiteConfig
from pochitrain.benchmark.utils import now_local_timestamp

LOGGER = logging.getLogger("pochitrain.benchmark")


def _resolve_config_path(model_path: Path) -> Path:
    """モデルパスから対応する config.py パスを解決する.

    Args:
        model_path: モデルファイルパス.

    Returns:
        推定した config.py パス.
    """
    return model_path.parent.parent / "config.py"


def _copy_case_config(case: CaseConfig, run_dir: Path) -> None:
    """ケースで使用する config.py を run ディレクトリへコピーする.

    Args:
        case: ケース設定.
        run_dir: ベンチマーク実行ディレクトリ.
    """
    source_config = _resolve_config_path(case.model_path)
    if not source_config.exists():
        LOGGER.warning(
            "config.py が見つからないためコピーをスキップします: case=%s path=%s",
            case.name,
            source_config,
        )
        return

    configs_dir = run_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    destination = configs_dir / f"{case.name}_config.py"
    shutil.copy2(source_config, destination)
    LOGGER.debug("config.py をコピーしました: %s", destination)


def _build_command(case: CaseConfig, run_output_dir: Path) -> List[str]:
    """ケース実行用コマンドを構築する.

    Args:
        case: ケース設定.
        run_output_dir: この試行の出力先.

    Returns:
        実行コマンド配列.
    """
    if case.runtime == "onnx":
        command = [
            sys.executable,
            "-m",
            "pochitrain.cli.infer_onnx",
            str(case.model_path),
            "--pipeline",
            case.pipeline,
            "--output",
            str(run_output_dir),
            "--benchmark-json",
        ]
    elif case.runtime == "trt":
        command = [
            sys.executable,
            "-m",
            "pochitrain.cli.infer_trt",
            str(case.model_path),
            "--pipeline",
            case.pipeline,
            "--output",
            str(run_output_dir),
            "--benchmark-json",
        ]
    elif case.runtime == "pytorch":
        command = [
            sys.executable,
            "-m",
            "pochitrain.cli.pochi",
            "infer",
            str(case.model_path),
            "--pipeline",
            case.pipeline,
            "--output",
            str(run_output_dir),
            "--benchmark-json",
        ]
    else:  # pragma: no cover
        raise ValueError(f"未対応runtimeです: {case.runtime}")

    if case.benchmark_env_name:
        command.extend(["--benchmark-env-name", case.benchmark_env_name])
    return command


def run_suite(suite: SuiteConfig, output_root: Path, fail_fast: bool) -> Path:
    """スイートを実行する.

    Args:
        suite: 実行対象スイート.
        output_root: ベンチマーク結果のルートディレクトリ.
        fail_fast: 失敗時に即終了するかどうか.

    Returns:
        作成した run ディレクトリ.
    """
    run_dir = output_root / f"{suite.name}_{now_local_timestamp()}"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    runtime_width = max(len(case.runtime) for case in suite.cases)
    pipeline_width = max(len(case.pipeline) for case in suite.cases)

    for case_index, case in enumerate(suite.cases, start=1):
        case_dir = raw_dir / f"case_{case_index:03d}_{case.name}"
        case_dir.mkdir(parents=True, exist_ok=True)
        _copy_case_config(case, run_dir)
        for repeat_index in range(1, case.repeats + 1):
            run_output_dir = case_dir / f"run_{repeat_index:03d}"
            run_output_dir.mkdir(parents=True, exist_ok=True)
            command = _build_command(case, run_output_dir)
            LOGGER.info(
                "suite=%s runtime=%s pipeline=%s repeat=%s/%s",
                suite.name,
                case.runtime.ljust(runtime_width),
                case.pipeline.ljust(pipeline_width),
                repeat_index,
                case.repeats,
            )
            LOGGER.debug("case=%s command=%s", case.name, " ".join(command))

            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )

            benchmark_json_path = run_output_dir / "benchmark_result.json"
            status = "ok"
            error: str | None = None
            if completed.returncode != 0:
                status = "failed"
                error = f"CLI returned non-zero exit code: {completed.returncode}"
            elif not benchmark_json_path.exists():
                status = "failed"
                error = "benchmark_result.json が生成されませんでした."

            if status != "ok":
                LOGGER.warning(
                    "実行失敗 suite=%s runtime=%s pipeline=%s repeat=%s/%s error=%s",
                    suite.name,
                    case.runtime.ljust(runtime_width),
                    case.pipeline.ljust(pipeline_width),
                    repeat_index,
                    case.repeats,
                    error,
                )
                if fail_fast:
                    raise RuntimeError(error)
    return run_dir
