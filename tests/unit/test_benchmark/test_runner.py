"""benchmark/runner.py のテスト."""

import sys
from pathlib import Path

from pochitrain.benchmark.models import CaseConfig, SuiteConfig
from pochitrain.benchmark.runner import (
    _build_command,
    _copy_case_config,
    _resolve_config_path,
)


def _make_case(
    *,
    name: str = "test_case",
    runtime: str = "onnx",
    model_path: Path = Path("/work/models/resnet18.onnx"),
    pipeline: str = "gpu",
    repeats: int = 1,
    benchmark_env_name: str | None = None,
) -> CaseConfig:
    """テスト用 CaseConfig を生成する."""
    return CaseConfig(
        name=name,
        runtime=runtime,
        model_path=model_path,
        pipeline=pipeline,
        repeats=repeats,
        benchmark_env_name=benchmark_env_name,
    )


class TestResolveConfigPath:
    """_resolve_config_path のテスト."""

    def test_resolves_to_parent_parent_config(self):
        """model_path の2階層上の config.py を返す."""
        model_path = Path("/work_dirs/20260316/models/best.pth")
        result = _resolve_config_path(model_path)
        assert result == Path("/work_dirs/20260316/config.py")

    def test_onnx_model_path(self):
        """ONNX モデルパスでも正しく解決する."""
        model_path = Path("/exports/onnx/resnet18.onnx")
        result = _resolve_config_path(model_path)
        assert result == Path("/exports/config.py")


class TestCopyCaseConfig:
    """_copy_case_config のテスト."""

    def test_copies_existing_config(self, tmp_path: Path):
        """config.py が存在する場合にコピーする."""
        # config.py を作成
        model_dir = tmp_path / "work_dirs" / "exp" / "models"
        model_dir.mkdir(parents=True)
        config_path = tmp_path / "work_dirs" / "exp" / "config.py"
        config_path.write_text("config = {}", encoding="utf-8")

        model_path = model_dir / "best.pth"
        model_path.touch()

        case = _make_case(model_path=model_path)
        run_dir = tmp_path / "benchmark_run"
        run_dir.mkdir()

        _copy_case_config(case, run_dir)

        copied = run_dir / "configs" / f"{case.name}_config.py"
        assert copied.exists()
        assert copied.read_text(encoding="utf-8") == "config = {}"

    def test_skips_when_config_missing(self, tmp_path: Path):
        """config.py が存在しない場合はスキップする."""
        model_path = tmp_path / "nonexistent" / "models" / "best.pth"
        case = _make_case(model_path=model_path)
        run_dir = tmp_path / "benchmark_run"
        run_dir.mkdir()

        # エラーが発生しないことを確認
        _copy_case_config(case, run_dir)

        configs_dir = run_dir / "configs"
        assert not configs_dir.exists()


class TestBuildCommand:
    """_build_command のテスト."""

    def test_onnx_command(self, tmp_path: Path):
        """ONNX ランタイムのコマンドを構築する."""
        case = _make_case(runtime="onnx", model_path=Path("/model.onnx"))
        output_dir = tmp_path / "output"

        cmd = _build_command(case, output_dir)

        assert cmd[0] == sys.executable
        assert cmd[1:3] == ["-m", "pochitrain.cli.infer_onnx"]
        assert str(Path("/model.onnx")) in cmd
        assert "--pipeline" in cmd
        assert "gpu" in cmd
        assert "--benchmark-json" in cmd

    def test_trt_command(self, tmp_path: Path):
        """TensorRT ランタイムのコマンドを構築する."""
        case = _make_case(runtime="trt", model_path=Path("/model.engine"))
        output_dir = tmp_path / "output"

        cmd = _build_command(case, output_dir)

        assert cmd[1:3] == ["-m", "pochitrain.cli.infer_trt"]

    def test_pytorch_command(self, tmp_path: Path):
        """PyTorch ランタイムのコマンドを構築する."""
        case = _make_case(runtime="pytorch", model_path=Path("/model.pth"))
        output_dir = tmp_path / "output"

        cmd = _build_command(case, output_dir)

        assert cmd[1:3] == ["-m", "pochitrain.cli.pochi"]
        assert "infer" in cmd

    def test_benchmark_env_name_appended(self, tmp_path: Path):
        """benchmark_env_name がコマンドに追加される."""
        case = _make_case(benchmark_env_name="windows11")
        output_dir = tmp_path / "output"

        cmd = _build_command(case, output_dir)

        idx = cmd.index("--benchmark-env-name")
        assert cmd[idx + 1] == "windows11"

    def test_no_env_name_not_in_command(self, tmp_path: Path):
        """benchmark_env_name が None の場合コマンドに含まれない."""
        case = _make_case(benchmark_env_name=None)
        output_dir = tmp_path / "output"

        cmd = _build_command(case, output_dir)

        assert "--benchmark-env-name" not in cmd

    def test_output_dir_in_command(self, tmp_path: Path):
        """出力ディレクトリがコマンドに含まれる."""
        case = _make_case()
        output_dir = tmp_path / "output"

        cmd = _build_command(case, output_dir)

        assert str(output_dir) in cmd
