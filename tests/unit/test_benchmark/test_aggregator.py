"""benchmark/aggregator.py のテスト."""

import json
from pathlib import Path

import pytest

from pochitrain.benchmark.aggregator import (
    _collect_benchmark_paths,
    _extract_case_name,
    aggregate_results,
)


def _write_benchmark_json(path: Path, data: dict) -> Path:
    """ヘルパー: benchmark_result.json を書き出す."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return path


def _make_result(
    *,
    env_name: str = "test_env",
    runtime: str = "onnx",
    precision: str = "fp32",
    model_name: str = "resnet18",
    pipeline: str = "gpu",
    device: str = "cuda",
    avg_inference_ms: float = 10.0,
    avg_e2e_ms: float = 15.0,
    accuracy_percent: float = 95.0,
) -> dict:
    """ベンチマーク結果の辞書を生成する."""
    return {
        "env_name": env_name,
        "runtime": runtime,
        "precision": precision,
        "model_name": model_name,
        "pipeline": pipeline,
        "device": device,
        "metrics": {
            "avg_inference_ms": avg_inference_ms,
            "avg_e2e_ms": avg_e2e_ms,
            "accuracy_percent": accuracy_percent,
        },
    }


class TestCollectBenchmarkPaths:
    """_collect_benchmark_paths のテスト."""

    def test_finds_files_in_raw_subdirectory(self, tmp_path: Path):
        """raw/ 配下の benchmark_result.json を見つける."""
        result_path = (
            tmp_path / "raw" / "case_001_test" / "run_001" / "benchmark_result.json"
        )
        _write_benchmark_json(result_path, {})

        paths = _collect_benchmark_paths(tmp_path)
        assert len(paths) == 1
        assert paths[0] == result_path

    def test_finds_files_without_raw(self, tmp_path: Path):
        """raw/ がない場合は input_dir 直下を検索する."""
        result_path = tmp_path / "case_001_test" / "run_001" / "benchmark_result.json"
        _write_benchmark_json(result_path, {})

        paths = _collect_benchmark_paths(tmp_path)
        assert len(paths) == 1

    def test_empty_directory_returns_empty(self, tmp_path: Path):
        """空ディレクトリで空リストを返す."""
        assert _collect_benchmark_paths(tmp_path) == []


class TestExtractCaseName:
    """_extract_case_name のテスト."""

    def test_standard_path(self, tmp_path: Path):
        """case_001_resnet18 から resnet18 を抽出する."""
        path = tmp_path / "case_001_resnet18" / "run_001" / "benchmark_result.json"
        assert _extract_case_name(path) == "resnet18"

    def test_case_name_with_underscores(self, tmp_path: Path):
        """アンダースコアを含むケース名を抽出する."""
        path = tmp_path / "case_002_resnet18_gpu" / "run_001" / "benchmark_result.json"
        assert _extract_case_name(path) == "resnet18_gpu"

    def test_non_case_prefix_returns_empty(self, tmp_path: Path):
        """case_ プレフィックスがない場合は空文字を返す."""
        path = tmp_path / "other_dir" / "run_001" / "benchmark_result.json"
        assert _extract_case_name(path) == ""


class TestAggregateResults:
    """aggregate_results のテスト."""

    def test_single_result(self, tmp_path: Path):
        """単一結果の集計."""
        result_path = (
            tmp_path / "raw" / "case_001_resnet18" / "run_001" / "benchmark_result.json"
        )
        _write_benchmark_json(result_path, _make_result())

        csv_path, json_path = aggregate_results(tmp_path)

        assert csv_path.exists()
        assert json_path.exists()
        assert csv_path.name == "benchmark_summary.csv"
        assert json_path.name == "benchmark_summary.json"

    def test_multiple_runs_aggregated(self, tmp_path: Path):
        """同一グループの複数ランが平均化される."""
        for i in range(1, 4):
            result_path = (
                tmp_path
                / "raw"
                / "case_001_resnet18"
                / f"run_{i:03d}"
                / "benchmark_result.json"
            )
            _write_benchmark_json(
                result_path,
                _make_result(avg_inference_ms=float(i * 10)),
            )

        csv_path, json_path = aggregate_results(tmp_path)

        with open(json_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        assert summary["total_benchmark_files"] == 3
        assert summary["total_groups"] == 1
        group = summary["groups"][0]
        assert group["runs"] == 3
        # mean of 10, 20, 30 = 20
        assert group["avg_inference_ms_mean"] == pytest.approx(20.0)

    def test_different_groups_separated(self, tmp_path: Path):
        """異なるグループが別々に集計される."""
        path1 = tmp_path / "raw" / "case_001_onnx" / "run_001" / "benchmark_result.json"
        _write_benchmark_json(path1, _make_result(runtime="onnx"))

        path2 = tmp_path / "raw" / "case_002_trt" / "run_001" / "benchmark_result.json"
        _write_benchmark_json(path2, _make_result(runtime="trt"))

        _, json_path = aggregate_results(tmp_path)

        with open(json_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        assert summary["total_groups"] == 2

    def test_no_results_raises(self, tmp_path: Path):
        """結果がない場合 ValueError."""
        with pytest.raises(ValueError, match="見つかりません"):
            aggregate_results(tmp_path)

    def test_invalid_json_counted(self, tmp_path: Path):
        """不正な JSON ファイルが invalid_files に記録される."""
        # 不正な JSON
        invalid_path = (
            tmp_path / "raw" / "case_001_bad" / "run_001" / "benchmark_result.json"
        )
        invalid_path.parent.mkdir(parents=True, exist_ok=True)
        invalid_path.write_text("not json", encoding="utf-8")

        # 有効な JSON (aggregate_results が ValueError にならないように)
        valid_path = (
            tmp_path / "raw" / "case_002_good" / "run_001" / "benchmark_result.json"
        )
        _write_benchmark_json(valid_path, _make_result())

        _, json_path = aggregate_results(tmp_path)

        with open(json_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        assert len(summary["invalid_files"]) == 1
        assert summary["total_groups"] == 1

    def test_csv_has_correct_headers(self, tmp_path: Path):
        """CSV に期待するヘッダーが含まれる."""
        result_path = (
            tmp_path / "raw" / "case_001_test" / "run_001" / "benchmark_result.json"
        )
        _write_benchmark_json(result_path, _make_result())

        csv_path, _ = aggregate_results(tmp_path)

        header_line = csv_path.read_text(encoding="utf-8").splitlines()[0]
        expected_headers = [
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
        assert header_line == ",".join(expected_headers)

    def test_stdev_zero_for_single_run(self, tmp_path: Path):
        """単一ランの標準偏差は 0."""
        result_path = (
            tmp_path / "raw" / "case_001_test" / "run_001" / "benchmark_result.json"
        )
        _write_benchmark_json(result_path, _make_result())

        _, json_path = aggregate_results(tmp_path)

        with open(json_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        group = summary["groups"][0]
        assert group["avg_inference_ms_stdev"] == 0.0
        assert group["avg_e2e_ms_stdev"] == 0.0
