"""infer_trt CLIの入口導線テスト."""

from unittest.mock import patch

import pytest

from pochitrain.cli.infer_trt import PIPELINE_CHOICES, main


def test_pipeline_choices_are_expected() -> None:
    """公開されるパイプライン候補が期待どおりであることを確認する."""
    assert PIPELINE_CHOICES == ("auto", "current", "fast", "gpu")


def test_main_no_args_exits() -> None:
    """引数なしでは argparse により SystemExit する."""
    with patch("sys.argv", ["infer-trt"]):
        with pytest.raises(SystemExit):
            main()


def test_main_nonexistent_engine_exits(tmp_path) -> None:
    """存在しないエンジンパス指定で SystemExit する."""
    fake_engine = str(tmp_path / "nonexistent.engine")
    with patch("sys.argv", ["infer-trt", fake_engine]):
        with pytest.raises(SystemExit):
            main()
