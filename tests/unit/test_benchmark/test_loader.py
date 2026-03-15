"""benchmark/loader.py のテスト."""

from pathlib import Path

import pytest
import yaml

from pochitrain.benchmark.loader import (
    _parse_model_paths,
    _parse_pipelines,
    _parse_positive_int,
    _parse_runtime,
    _require_non_empty_str,
    load_suite_config,
)


def _write_yaml(path: Path, data: dict) -> Path:
    """ヘルパー: YAML ファイルを書き出す."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False)
    return path


# --- バリデーション関数のテスト ---


class TestRequireNonEmptyStr:
    """_require_non_empty_str のテスト."""

    def test_valid_string(self):
        """通常の文字列を返す."""
        assert _require_non_empty_str("hello", "field") == "hello"

    def test_strips_whitespace(self):
        """前後の空白を除去する."""
        assert _require_non_empty_str("  hello  ", "field") == "hello"

    def test_empty_string_raises(self):
        """空文字列で ValueError."""
        with pytest.raises(ValueError, match="空でない文字列"):
            _require_non_empty_str("", "field")

    def test_whitespace_only_raises(self):
        """空白のみで ValueError."""
        with pytest.raises(ValueError, match="空でない文字列"):
            _require_non_empty_str("   ", "field")

    def test_non_string_raises(self):
        """文字列以外で ValueError."""
        with pytest.raises(ValueError, match="空でない文字列"):
            _require_non_empty_str(123, "field")


class TestParsePositiveInt:
    """_parse_positive_int のテスト."""

    def test_valid_positive(self):
        """正の整数を返す."""
        assert _parse_positive_int(5, "repeats") == 5

    def test_zero_raises(self):
        """0 で ValueError."""
        with pytest.raises(ValueError, match="正の整数"):
            _parse_positive_int(0, "repeats")

    def test_negative_raises(self):
        """負数で ValueError."""
        with pytest.raises(ValueError, match="正の整数"):
            _parse_positive_int(-1, "repeats")

    def test_float_raises(self):
        """float で ValueError."""
        with pytest.raises(ValueError, match="正の整数"):
            _parse_positive_int(1.5, "repeats")


class TestParseRuntime:
    """_parse_runtime のテスト."""

    def test_onnx(self):
        """onnx を受け付ける."""
        assert _parse_runtime("onnx", "runtime") == "onnx"

    def test_trt(self):
        """trt を受け付ける."""
        assert _parse_runtime("trt", "runtime") == "trt"

    def test_pytorch(self):
        """pytorch を受け付ける."""
        assert _parse_runtime("pytorch", "runtime") == "pytorch"

    def test_case_insensitive(self):
        """大文字でも受け付ける."""
        assert _parse_runtime("ONNX", "runtime") == "onnx"

    def test_invalid_runtime_raises(self):
        """不正な runtime で ValueError."""
        with pytest.raises(ValueError, match="onnx, trt, pytorch"):
            _parse_runtime("tensorflow", "runtime")


class TestParsePipelines:
    """_parse_pipelines のテスト."""

    def test_single_string(self):
        """単一文字列をリストに変換する."""
        assert _parse_pipelines("gpu", "pipelines") == ["gpu"]

    def test_list_of_strings(self):
        """リストをそのまま返す."""
        assert _parse_pipelines(["gpu", "fast"], "pipelines") == ["gpu", "fast"]

    def test_deduplication(self):
        """重複を除去する."""
        assert _parse_pipelines(["gpu", "gpu", "fast"], "pipelines") == ["gpu", "fast"]

    def test_invalid_pipeline_raises(self):
        """不正な pipeline で ValueError."""
        with pytest.raises(ValueError, match="いずれかを指定してください"):
            _parse_pipelines(["invalid"], "pipelines")

    def test_empty_list_raises(self):
        """空リストで ValueError."""
        with pytest.raises(ValueError, match="1件以上必要"):
            _parse_pipelines([], "pipelines")

    def test_non_list_non_string_raises(self):
        """リストでも文字列でもない場合に ValueError."""
        with pytest.raises(ValueError, match="文字列または文字列リスト"):
            _parse_pipelines(123, "pipelines")


class TestParseModelPaths:
    """_parse_model_paths のテスト."""

    def test_valid_mapping(self):
        """runtime から Path への辞書を返す."""
        result = _parse_model_paths(
            {"onnx": "/model.onnx", "trt": "/model.engine"}, "model_paths"
        )
        assert result == {
            "onnx": Path("/model.onnx"),
            "trt": Path("/model.engine"),
        }

    def test_none_returns_empty(self):
        """None で空辞書を返す."""
        assert _parse_model_paths(None, "model_paths") == {}

    def test_non_dict_raises(self):
        """辞書以外で ValueError."""
        with pytest.raises(ValueError, match="辞書"):
            _parse_model_paths("not_a_dict", "model_paths")

    def test_invalid_runtime_key_raises(self):
        """不正な runtime キーで ValueError."""
        with pytest.raises(ValueError, match="onnx, trt, pytorch"):
            _parse_model_paths({"invalid": "/model"}, "model_paths")


# --- load_suite_config のテスト ---


class TestLoadSuiteConfig:
    """load_suite_config のテスト."""

    def test_minimal_suite(self, tmp_path: Path):
        """最小構成のスイートを正しく読み込む."""
        data = {
            "suites": {
                "base": {
                    "cases": [
                        {
                            "name": "resnet18",
                            "runtime": "onnx",
                            "model_path": "/models/resnet18.onnx",
                            "pipeline": "gpu",
                        }
                    ]
                }
            }
        }
        yaml_path = _write_yaml(tmp_path / "suites.yaml", data)
        suite = load_suite_config(yaml_path, "base")

        assert suite.name == "base"
        assert len(suite.cases) == 1
        assert suite.cases[0].runtime == "onnx"
        assert suite.cases[0].pipeline == "gpu"
        assert suite.cases[0].repeats == 1

    def test_suite_with_defaults(self, tmp_path: Path):
        """defaults からパイプラインとモデルパスを継承する."""
        data = {
            "suites": {
                "full": {
                    "description": "Full benchmark",
                    "repeats": 3,
                    "defaults": {
                        "pipelines": ["gpu", "fast"],
                        "benchmark_env_name": "windows11",
                        "model_paths": {
                            "onnx": "/default.onnx",
                        },
                    },
                    "cases": [
                        {
                            "name": "resnet18",
                            "runtime": "onnx",
                        }
                    ],
                }
            }
        }
        yaml_path = _write_yaml(tmp_path / "suites.yaml", data)
        suite = load_suite_config(yaml_path, "full")

        assert suite.description == "Full benchmark"
        # 2 pipelines x 1 case = 2 CaseConfig
        assert len(suite.cases) == 2
        assert suite.cases[0].pipeline == "gpu"
        assert suite.cases[1].pipeline == "fast"
        assert suite.cases[0].repeats == 3
        assert suite.cases[0].benchmark_env_name == "windows11"
        assert suite.cases[0].model_path == Path("/default.onnx")

    def test_case_overrides_defaults(self, tmp_path: Path):
        """case レベルの指定が defaults を上書きする."""
        data = {
            "suites": {
                "test": {
                    "defaults": {
                        "pipelines": ["gpu"],
                        "model_paths": {"onnx": "/default.onnx"},
                    },
                    "cases": [
                        {
                            "name": "custom",
                            "runtime": "onnx",
                            "model_path": "/custom.onnx",
                            "pipeline": "fast",
                        }
                    ],
                }
            }
        }
        yaml_path = _write_yaml(tmp_path / "suites.yaml", data)
        suite = load_suite_config(yaml_path, "test")

        assert suite.cases[0].model_path == Path("/custom.onnx")
        assert suite.cases[0].pipeline == "fast"

    def test_auto_generated_name(self, tmp_path: Path):
        """name 未指定時に自動生成される."""
        data = {
            "suites": {
                "test": {
                    "cases": [
                        {
                            "runtime": "trt",
                            "model_path": "/model.engine",
                            "pipeline": "gpu",
                        }
                    ]
                }
            }
        }
        yaml_path = _write_yaml(tmp_path / "suites.yaml", data)
        suite = load_suite_config(yaml_path, "test")

        # name 未指定 → "{runtime}_{index:02d}_{pipeline}" 形式
        assert "trt" in suite.cases[0].name
        assert "gpu" in suite.cases[0].name

    def test_duplicate_name_gets_suffix(self, tmp_path: Path):
        """重複名が発生すると index サフィックスが付与される."""
        data = {
            "suites": {
                "test": {
                    "defaults": {"pipelines": ["gpu"]},
                    "cases": [
                        {
                            "runtime": "onnx",
                            "model_path": "/m.onnx",
                            "pipeline": "gpu",
                        },
                        {
                            "runtime": "onnx",
                            "model_path": "/m2.onnx",
                            "pipeline": "gpu",
                        },
                    ],
                }
            }
        }
        yaml_path = _write_yaml(tmp_path / "suites.yaml", data)
        suite = load_suite_config(yaml_path, "test")

        names = [c.name for c in suite.cases]
        assert len(names) == len(set(names)), "名前が重複している"

    def test_nonexistent_file_raises(self, tmp_path: Path):
        """存在しないファイルで FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="見つかりません"):
            load_suite_config(tmp_path / "nonexistent.yaml", "base")

    def test_invalid_root_raises(self, tmp_path: Path):
        """ルートが辞書でない YAML で ValueError."""
        yaml_path = tmp_path / "suites.yaml"
        yaml_path.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="ルートは辞書"):
            load_suite_config(yaml_path, "base")

    def test_missing_suites_section_raises(self, tmp_path: Path):
        """suites セクションがない YAML で ValueError."""
        yaml_path = _write_yaml(tmp_path / "suites.yaml", {"other": "data"})
        with pytest.raises(ValueError, match="suites セクション"):
            load_suite_config(yaml_path, "base")

    def test_unknown_suite_name_raises(self, tmp_path: Path):
        """存在しないスイート名で ValueError."""
        data: dict = {"suites": {"existing": {"cases": []}}}
        yaml_path = _write_yaml(tmp_path / "suites.yaml", data)
        with pytest.raises(ValueError, match="見つからないか不正"):
            load_suite_config(yaml_path, "nonexistent")

    def test_empty_cases_raises(self, tmp_path: Path):
        """cases が空で ValueError."""
        data: dict = {"suites": {"test": {"cases": []}}}
        yaml_path = _write_yaml(tmp_path / "suites.yaml", data)
        with pytest.raises(ValueError, match="1件以上のリスト"):
            load_suite_config(yaml_path, "test")

    def test_missing_model_path_raises(self, tmp_path: Path):
        """model_path が解決できない場合 ValueError."""
        data = {
            "suites": {
                "test": {
                    "cases": [
                        {
                            "name": "no_model",
                            "runtime": "onnx",
                            "pipeline": "gpu",
                        }
                    ]
                }
            }
        }
        yaml_path = _write_yaml(tmp_path / "suites.yaml", data)
        with pytest.raises(ValueError, match="model_path が未指定"):
            load_suite_config(yaml_path, "test")
