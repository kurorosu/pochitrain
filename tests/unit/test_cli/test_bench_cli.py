"""`bench` CLI のテスト.
成功系は実際に CLI を呼び出してベンチマークを実行し、失敗系は引数エラーやファイルエラーなどを検証する.
"""

import json
from pathlib import Path

from pochitrain.cli.bench import main, parse_args


class TestParseArgs:
    """引数解析テスト."""

    def test_defaults(self):
        """デフォルト値を検証する."""
        args = parse_args([])
        assert args.suite is None
        assert args.suites_file == "configs/bench_suites.yaml"
        assert args.aggregate_only is False
        assert args.input_dir is None
        assert args.fail_fast is False
        assert args.debug is False

    def test_suite_option(self):
        """--suite 指定値を検証する."""
        args = parse_args(["--suite", "base"])
        assert args.suite == "base"

    def test_suites_file_option(self):
        """--suites-file 指定値を検証する."""
        args = parse_args(["--suites-file", "/tmp/custom.yaml"])
        assert args.suites_file == "/tmp/custom.yaml"

    def test_aggregate_only_with_input_dir(self):
        """--aggregate-only と --input-dir の組み合わせを検証する."""
        args = parse_args(["--aggregate-only", "--input-dir", "/tmp/results"])
        assert args.aggregate_only is True
        assert args.input_dir == "/tmp/results"

    def test_fail_fast_and_debug(self):
        """--fail-fast と --debug フラグを検証する."""
        args = parse_args(["--fail-fast", "--debug"])
        assert args.fail_fast is True
        assert args.debug is True


class TestMainAggregateOnly:
    """--aggregate-only モードのテスト."""

    def test_aggregate_only_requires_input_dir(self):
        """--aggregate-only で --input-dir 未指定は戻り値 1."""
        assert main(["--aggregate-only"]) == 1

    def test_aggregate_only_success(self, tmp_path: Path):
        """集計成功パスで summary/ が生成されることを検証する."""
        case_dir = tmp_path / "raw" / "case_001_test" / "run_001"
        case_dir.mkdir(parents=True)
        benchmark_result = {
            "env_name": "test_env",
            "runtime": "onnx",
            "precision": "fp32",
            "model_name": "resnet18",
            "pipeline": "gpu",
            "device": "cuda",
            "metrics": {
                "avg_inference_ms": 10.5,
                "avg_e2e_ms": 15.2,
                "accuracy_percent": 95.0,
            },
        }
        result_file = case_dir / "benchmark_result.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(benchmark_result, f)

        exit_code = main(["--aggregate-only", "--input-dir", str(tmp_path)])
        assert exit_code == 0
        summary_dir = tmp_path / "summary"
        assert summary_dir.exists()
        assert (summary_dir / "benchmark_summary.csv").exists()
        assert (summary_dir / "benchmark_summary.json").exists()


class TestMainRunMode:
    """実行モードのテスト."""

    def test_run_mode_requires_suite(self):
        """--suite 未指定は戻り値 1."""
        assert main([]) == 1

    def test_nonexistent_suites_file(self):
        """存在しない suites.yaml パスで戻り値 1."""
        assert main(["--suite", "base", "--suites-file", "/nonexistent.yaml"]) == 1
