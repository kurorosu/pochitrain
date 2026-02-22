"""ベンチマークスイート設定の読み込み."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml  # type: ignore[import-untyped]
from models import CaseConfig, SuiteConfig


def _require_non_empty_str(value: Any, field: str) -> str:
    """必須文字列を検証する.

    Args:
        value: 検証対象値.
        field: エラー表示用フィールド名.

    Returns:
        検証済み文字列.

    Raises:
        ValueError: 文字列でないか空文字の場合.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} は空でない文字列である必要があります.")
    return value.strip()


def _parse_optional_str(value: Any, field: str) -> Optional[str]:
    """任意文字列を検証する.

    Args:
        value: 検証対象値.
        field: エラー表示用フィールド名.

    Returns:
        検証済み文字列. 未指定時は None.
    """
    if value is None:
        return None
    return _require_non_empty_str(value, field)


def _parse_positive_int(value: Any, field: str) -> int:
    """正の整数を検証する.

    Args:
        value: 検証対象値.
        field: エラー表示用フィールド名.

    Returns:
        検証済み整数.

    Raises:
        ValueError: 正の整数でない場合.
    """
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} は正の整数である必要があります: {value}")
    return value


def _parse_runtime(value: Any, field: str) -> str:
    """Runtime を検証する.

    Args:
        value: 検証対象値.
        field: エラー表示用フィールド名.

    Returns:
        `onnx` または `trt`.

    Raises:
        ValueError: 不正な runtime の場合.
    """
    runtime = _require_non_empty_str(value, field).lower()
    if runtime not in {"onnx", "trt"}:
        raise ValueError(f"{field} は onnx または trt を指定してください: {value}")
    return runtime


def _parse_pipelines(value: Any, field: str) -> List[str]:
    """Pipeline 指定をリスト形式へ正規化する.

    Args:
        value: 検証対象値. 文字列または文字列リストを受け付ける.
        field: エラー表示用フィールド名.

    Returns:
        検証済み pipeline 名リスト.
    """
    if isinstance(value, str):
        raw_items = [value]
    elif isinstance(value, list):
        raw_items = value
    else:
        raise ValueError(f"{field} は文字列または文字列リストである必要があります.")

    allowed = {"auto", "current", "fast", "gpu"}
    parsed: List[str] = []
    for index, item in enumerate(raw_items):
        pipeline = _require_non_empty_str(item, f"{field}[{index}]")
        if pipeline not in allowed:
            raise ValueError(
                f"{field}[{index}] は {sorted(allowed)} のいずれかを指定してください: {pipeline}"
            )
        if pipeline not in parsed:
            parsed.append(pipeline)
    if len(parsed) == 0:
        raise ValueError(f"{field} は1件以上必要です.")
    return parsed


def _parse_model_paths(value: Any, field: str) -> Dict[str, Path]:
    """Runtime ごとの model_path マップを検証する.

    Args:
        value: 検証対象値.
        field: エラー表示用フィールド名.

    Returns:
        runtime から model_path への辞書.
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field} は辞書である必要があります.")

    result: Dict[str, Path] = {}
    for key, raw_path in value.items():
        runtime = _parse_runtime(key, f"{field}.key")
        path_text = _require_non_empty_str(raw_path, f"{field}.{runtime}")
        result[runtime] = Path(path_text)
    return result


def load_suite_config(suites_file: Path, suite_name: str) -> SuiteConfig:
    """指定スイートを YAML から読み込む.

    Args:
        suites_file: suites.yaml パス.
        suite_name: スイート名.

    Returns:
        検証済みスイート設定.
    """
    if not suites_file.exists():
        raise FileNotFoundError(f"suites.yaml が見つかりません: {suites_file}")

    with open(suites_file, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    if not isinstance(payload, dict):
        raise ValueError("suites.yaml のルートは辞書である必要があります.")
    suites = payload.get("suites")
    if not isinstance(suites, dict):
        raise ValueError("suites.yaml に suites セクションが必要です.")
    raw_suite = suites.get(suite_name)
    if not isinstance(raw_suite, dict):
        raise ValueError(f"スイート定義が見つからないか不正です: {suite_name}")

    description = str(raw_suite.get("description", "")).strip()
    suite_repeats = _parse_positive_int(
        raw_suite.get("repeats", 1), f"{suite_name}.repeats"
    )

    raw_defaults = raw_suite.get("defaults", {})
    if raw_defaults is None:
        raw_defaults = {}
    if not isinstance(raw_defaults, dict):
        raise ValueError(f"{suite_name}.defaults は辞書である必要があります.")

    default_pipelines = _parse_pipelines(
        raw_defaults.get("pipelines", raw_defaults.get("pipeline", "gpu")),
        f"{suite_name}.defaults.pipelines",
    )
    default_env_name = _parse_optional_str(
        raw_defaults.get("benchmark_env_name"),
        f"{suite_name}.defaults.benchmark_env_name",
    )
    default_model_paths = _parse_model_paths(
        raw_defaults.get("model_paths"),
        f"{suite_name}.defaults.model_paths",
    )

    raw_cases = raw_suite.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) == 0:
        raise ValueError(f"{suite_name}.cases は1件以上のリストである必要があります.")

    cases: List[CaseConfig] = []
    used_names: set[str] = set()
    for index, raw_case in enumerate(raw_cases, start=1):
        field_prefix = f"{suite_name}.cases[{index}]"
        if not isinstance(raw_case, dict):
            raise ValueError(f"{field_prefix} は辞書である必要があります.")

        runtime = _parse_runtime(raw_case.get("runtime"), f"{field_prefix}.runtime")
        raw_model_path = raw_case.get("model_path")
        if raw_model_path is None:
            model_path = default_model_paths.get(runtime)
        else:
            model_path = Path(
                _require_non_empty_str(raw_model_path, f"{field_prefix}.model_path")
            )
        if model_path is None:
            raise ValueError(
                f"{field_prefix}.model_path が未指定です. "
                "case.model_path または defaults.model_paths.<runtime> を指定してください."
            )

        case_pipelines = _parse_pipelines(
            raw_case.get("pipelines", raw_case.get("pipeline", default_pipelines)),
            f"{field_prefix}.pipelines",
        )
        repeats = _parse_positive_int(
            raw_case.get("repeats", suite_repeats),
            f"{field_prefix}.repeats",
        )
        env_name = _parse_optional_str(
            raw_case.get("benchmark_env_name"),
            f"{field_prefix}.benchmark_env_name",
        )
        if env_name is None:
            env_name = default_env_name

        raw_name = raw_case.get("name")
        if raw_name is None:
            base_name = f"{runtime}_{index:02d}"
        else:
            base_name = _require_non_empty_str(raw_name, f"{field_prefix}.name")

        for pipeline in case_pipelines:
            name = (
                f"{base_name}_{pipeline}"
                if (len(case_pipelines) > 1 or raw_name is None)
                else base_name
            )
            if name in used_names:
                name = f"{name}_{index:02d}"
            used_names.add(name)

            cases.append(
                CaseConfig(
                    name=name,
                    runtime=runtime,
                    model_path=model_path,
                    pipeline=pipeline,
                    repeats=repeats,
                    benchmark_env_name=env_name,
                )
            )

    return SuiteConfig(name=suite_name, description=description, cases=cases)
