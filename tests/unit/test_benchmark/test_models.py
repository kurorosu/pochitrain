"""benchmark/models.py のテスト."""

from pathlib import Path

from pochitrain.benchmark.models import CaseConfig, SuiteConfig


class TestCaseConfig:
    """CaseConfig のテスト."""

    def test_fields_are_stored(self):
        """全フィールドが正しく保持される."""
        case = CaseConfig(
            name="resnet18_gpu",
            runtime="onnx",
            model_path=Path("/models/resnet18.onnx"),
            pipeline="gpu",
            repeats=3,
            benchmark_env_name="windows11",
        )
        assert case.name == "resnet18_gpu"
        assert case.runtime == "onnx"
        assert case.model_path == Path("/models/resnet18.onnx")
        assert case.pipeline == "gpu"
        assert case.repeats == 3
        assert case.benchmark_env_name == "windows11"

    def test_benchmark_env_name_none(self):
        """benchmark_env_name が None でも生成できる."""
        case = CaseConfig(
            name="test",
            runtime="trt",
            model_path=Path("/model.engine"),
            pipeline="fast",
            repeats=1,
            benchmark_env_name=None,
        )
        assert case.benchmark_env_name is None

    def test_frozen(self):
        """frozen dataclass なので属性変更できない."""
        case = CaseConfig(
            name="test",
            runtime="pytorch",
            model_path=Path("/model.pth"),
            pipeline="current",
            repeats=1,
            benchmark_env_name=None,
        )
        try:
            case.name = "changed"  # type: ignore[misc]
            assert False, "FrozenInstanceError が発生するべき"
        except AttributeError:
            pass


class TestSuiteConfig:
    """SuiteConfig のテスト."""

    def test_suite_with_cases(self):
        """cases リスト付きで正しく生成される."""
        case = CaseConfig(
            name="case1",
            runtime="onnx",
            model_path=Path("/m.onnx"),
            pipeline="gpu",
            repeats=2,
            benchmark_env_name=None,
        )
        suite = SuiteConfig(name="base", description="Base suite", cases=[case])
        assert suite.name == "base"
        assert suite.description == "Base suite"
        assert len(suite.cases) == 1
        assert suite.cases[0].name == "case1"

    def test_empty_cases_list(self):
        """空の cases リストでも生成できる."""
        suite = SuiteConfig(name="empty", description="", cases=[])
        assert suite.cases == []
