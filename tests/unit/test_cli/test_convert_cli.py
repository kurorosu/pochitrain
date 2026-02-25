"""`pochi convert` CLIのテスト.

実際のエントリポイントを通して, 引数パースと変換実行の連携を検証する.
"""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import pochitrain.cli.pochi as pochi_cli
from pochitrain.cli.arg_types import positive_int

ConvertArgsRunner = Callable[[list[str]], argparse.Namespace]


@pytest.fixture
def convert_args(monkeypatch: pytest.MonkeyPatch) -> ConvertArgsRunner:
    """`pochi main` 経由で `convert` のパース結果を取得するfixture."""

    def _run(argv_tail: list[str]) -> argparse.Namespace:
        captured: dict[str, argparse.Namespace] = {}

        def _fake_convert(args: argparse.Namespace) -> None:
            captured["args"] = args

        monkeypatch.setattr(pochi_cli, "convert_command", _fake_convert)
        monkeypatch.setattr("sys.argv", ["pochi", *argv_tail])
        pochi_cli.main()
        return captured["args"]

    return _run


class TestConvertParser:
    """`convert` サブコマンドの引数パース検証."""

    def test_onnx_path_required(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`onnx_path` が必須引数であることを確認."""
        monkeypatch.setattr("sys.argv", ["pochi", "convert"])
        with pytest.raises(SystemExit):
            pochi_cli.main()

    def test_parse_basic_options(self, convert_args: ConvertArgsRunner) -> None:
        """主要オプションが想定どおりにパースされることを確認."""
        args = convert_args(
            [
                "convert",
                "model.onnx",
                "--fp16",
                "--config-path",
                "config.py",
                "--calib-data",
                "data/val",
                "--input-size",
                "224",
                "224",
                "--calib-samples",
                "200",
                "--calib-batch-size",
                "4",
                "--workspace-size",
                "1024",
                "--output",
                "out.engine",
            ]
        )

        assert args.command == "convert"
        assert args.onnx_path == "model.onnx"
        assert args.fp16 is True
        assert args.int8 is False
        assert args.config_path == "config.py"
        assert args.calib_data == "data/val"
        assert args.input_size == [224, 224]
        assert args.calib_samples == 200
        assert args.calib_batch_size == 4
        assert args.workspace_size == 1024
        assert args.output == "out.engine"

    def test_parse_defaults(self, convert_args: ConvertArgsRunner) -> None:
        """未指定時にデフォルト値が適用されることを確認."""
        args = convert_args(["convert", "model.onnx"])

        assert args.fp16 is False
        assert args.int8 is False
        assert args.input_size is None
        assert args.calib_samples == 500
        assert args.calib_batch_size == 1
        assert args.workspace_size == 1 << 30


class FakeConverter:
    """`TensorRTConverter` を置き換えるテスト用ダミー."""

    instances: list["FakeConverter"] = []
    latest_instance: "FakeConverter | None" = None

    def __init__(self, onnx_path: Path, workspace_size: int) -> None:
        """コンストラクタ引数と呼び出し履歴を保持."""
        self.onnx_path = onnx_path
        self.workspace_size = workspace_size
        self.convert_calls: list[dict] = []
        FakeConverter.instances.append(self)
        FakeConverter.latest_instance = self

    def convert(
        self,
        output_path: Path,
        precision: str,
        calibrator: object,
        input_shape: tuple[int, int, int] | None,
    ) -> Path:
        """変換呼び出し内容を記録し, 出力パスをそのまま返す."""
        self.convert_calls.append(
            {
                "output_path": output_path,
                "precision": precision,
                "calibrator": calibrator,
                "input_shape": input_shape,
            }
        )
        return output_path


class TestConvertCommand:
    """`convert_command` の実行時分岐と引数伝播を検証."""

    def setup_method(self) -> None:
        """テスト間でダミーインスタンスの履歴を初期化."""
        FakeConverter.instances.clear()
        FakeConverter.latest_instance = None

    def test_output_suffix_and_input_shape(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """デフォルト出力名のsuffixと `input_shape` 変換を確認."""
        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_text("dummy", encoding="utf-8")

        logger = SimpleNamespace(
            debug=lambda *_a, **_k: None,
            info=lambda *_a, **_k: None,
            error=lambda *_a, **_k: None,
        )

        import pochitrain.tensorrt.converter as converter_module
        import pochitrain.tensorrt.inference as inference_module

        monkeypatch.setattr(converter_module, "TensorRTConverter", FakeConverter)
        monkeypatch.setattr(
            inference_module, "check_tensorrt_availability", lambda: True
        )
        monkeypatch.setattr(pochi_cli, "setup_logging", lambda debug=False: logger)

        args = argparse.Namespace(
            debug=False,
            onnx_path=str(onnx_path),
            fp16=True,
            int8=False,
            output=None,
            config_path=None,
            calib_data=None,
            input_size=[224, 224],
            calib_samples=500,
            calib_batch_size=1,
            workspace_size=1 << 20,
        )

        pochi_cli.convert_command(args)

        instance = FakeConverter.latest_instance
        assert instance is not None
        assert instance.onnx_path == onnx_path
        assert instance.workspace_size == (1 << 20)
        call = instance.convert_calls[0]
        assert call["precision"] == "fp16"
        assert call["output_path"] == onnx_path.with_name("model_fp16.engine")
        assert call["input_shape"] == (3, 224, 224)

    def test_explicit_output_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`--output` 指定時は明示パスが優先されることを確認."""
        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_text("dummy", encoding="utf-8")
        explicit_output = tmp_path / "custom.engine"

        logger = SimpleNamespace(
            debug=lambda *_a, **_k: None,
            info=lambda *_a, **_k: None,
            error=lambda *_a, **_k: None,
        )

        import pochitrain.tensorrt.converter as converter_module
        import pochitrain.tensorrt.inference as inference_module

        monkeypatch.setattr(converter_module, "TensorRTConverter", FakeConverter)
        monkeypatch.setattr(
            inference_module, "check_tensorrt_availability", lambda: True
        )
        monkeypatch.setattr(pochi_cli, "setup_logging", lambda debug=False: logger)

        args = argparse.Namespace(
            debug=False,
            onnx_path=str(onnx_path),
            fp16=False,
            int8=False,
            output=str(explicit_output),
            config_path=None,
            calib_data=None,
            input_size=[128, 256],
            calib_samples=500,
            calib_batch_size=1,
            workspace_size=1 << 20,
        )

        pochi_cli.convert_command(args)

        instance = FakeConverter.latest_instance
        assert instance is not None
        assert instance.onnx_path == onnx_path
        assert instance.workspace_size == (1 << 20)
        call = instance.convert_calls[0]
        assert call["precision"] == "fp32"
        assert call["output_path"] == explicit_output
        assert call["input_shape"] == (3, 128, 256)

    def test_int8_takes_precedence_and_builds_calibrator(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`--int8` と `--fp16` 同時指定時にINT8が優先されることを確認."""
        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_text("dummy", encoding="utf-8")
        calib_dir = tmp_path / "calib"
        calib_dir.mkdir()

        logger = SimpleNamespace(
            debug=Mock(),
            info=Mock(),
            error=Mock(),
        )
        fake_calibrator = object()
        calibrator_args: dict[str, object] = {}

        def _fake_create_int8_calibrator(**kwargs: object) -> object:
            calibrator_args.update(kwargs)
            return fake_calibrator

        import pochitrain.tensorrt.calibrator as calibrator_module
        import pochitrain.tensorrt.converter as converter_module
        import pochitrain.tensorrt.inference as inference_module

        monkeypatch.setattr(converter_module, "TensorRTConverter", FakeConverter)
        monkeypatch.setattr(
            inference_module, "check_tensorrt_availability", lambda: True
        )
        monkeypatch.setattr(
            calibrator_module, "create_int8_calibrator", _fake_create_int8_calibrator
        )
        monkeypatch.setattr(
            pochi_cli.ConfigLoader,
            "load_config",
            lambda _path: {"val_transform": "dummy_transform"},
        )
        monkeypatch.setattr(pochi_cli, "setup_logging", lambda debug=False: logger)

        args = argparse.Namespace(
            debug=False,
            onnx_path=str(onnx_path),
            fp16=True,
            int8=True,
            output=None,
            config_path=str(tmp_path / "config.py"),
            calib_data=str(calib_dir),
            input_size=[224, 224],
            calib_samples=123,
            calib_batch_size=2,
            workspace_size=1 << 20,
        )

        pochi_cli.convert_command(args)

        instance = FakeConverter.latest_instance
        assert instance is not None
        assert instance.onnx_path == onnx_path
        assert instance.workspace_size == (1 << 20)
        call = instance.convert_calls[0]
        assert call["precision"] == "int8"
        assert call["output_path"] == onnx_path.with_name("model_int8.engine")
        assert call["calibrator"] is fake_calibrator
        assert call["input_shape"] == (3, 224, 224)

        assert calibrator_args["data_root"] == str(calib_dir)
        assert calibrator_args["input_shape"] == (3, 224, 224)
        assert calibrator_args["batch_size"] == 2
        assert calibrator_args["max_samples"] == 123
        error_messages = [
            str(call.args[0]) for call in logger.error.call_args_list if call.args
        ]
        assert not error_messages

    def test_int8_uses_val_data_root_from_auto_config(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """INT8で--calib-data未指定時に自動設定のval_data_rootを使うことを確認."""
        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_text("dummy", encoding="utf-8")
        calib_dir = tmp_path / "auto_calib"
        calib_dir.mkdir()

        logger = SimpleNamespace(
            debug=Mock(),
            info=Mock(),
            error=Mock(),
        )
        fake_calibrator = object()
        calibrator_args: dict[str, object] = {}

        def _fake_create_int8_calibrator(**kwargs: object) -> object:
            calibrator_args.update(kwargs)
            return fake_calibrator

        import pochitrain.tensorrt.calibrator as calibrator_module
        import pochitrain.tensorrt.converter as converter_module
        import pochitrain.tensorrt.inference as inference_module

        monkeypatch.setattr(converter_module, "TensorRTConverter", FakeConverter)
        monkeypatch.setattr(
            inference_module, "check_tensorrt_availability", lambda: True
        )
        monkeypatch.setattr(
            calibrator_module, "create_int8_calibrator", _fake_create_int8_calibrator
        )
        monkeypatch.setattr(
            pochi_cli,
            "load_config_auto",
            lambda _path: {
                "val_data_root": str(calib_dir),
                "val_transform": "dummy_transform",
            },
        )
        monkeypatch.setattr(pochi_cli, "setup_logging", lambda debug=False: logger)

        args = argparse.Namespace(
            debug=False,
            onnx_path=str(onnx_path),
            fp16=False,
            int8=True,
            output=None,
            config_path=None,
            calib_data=None,
            input_size=[256, 256],
            calib_samples=50,
            calib_batch_size=8,
            workspace_size=1 << 21,
        )

        pochi_cli.convert_command(args)

        instance = FakeConverter.latest_instance
        assert instance is not None
        assert instance.onnx_path == onnx_path
        assert instance.workspace_size == (1 << 21)
        call = instance.convert_calls[0]
        assert call["precision"] == "int8"
        assert call["calibrator"] is fake_calibrator
        assert call["input_shape"] == (3, 256, 256)
        assert calibrator_args["data_root"] == str(calib_dir)
        assert calibrator_args["batch_size"] == 8
        assert calibrator_args["max_samples"] == 50
        error_messages = [
            str(call.args[0]) for call in logger.error.call_args_list if call.args
        ]
        assert not error_messages

    def test_dynamic_onnx_requires_input_size(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """動的シェイプONNXで `--input-size` 未指定時にエラー終了することを確認."""
        onnx_path = tmp_path / "dynamic.onnx"
        onnx_path.write_text("dummy", encoding="utf-8")

        logger = SimpleNamespace(
            debug=Mock(),
            info=Mock(),
            error=Mock(),
        )

        import pochitrain.tensorrt.converter as converter_module
        import pochitrain.tensorrt.inference as inference_module

        monkeypatch.setattr(converter_module, "TensorRTConverter", FakeConverter)
        monkeypatch.setattr(
            inference_module, "check_tensorrt_availability", lambda: True
        )
        monkeypatch.setattr(pochi_cli, "setup_logging", lambda debug=False: logger)

        dynamic_onnx = SimpleNamespace(
            load=lambda _path: SimpleNamespace(
                graph=SimpleNamespace(
                    input=[
                        SimpleNamespace(
                            type=SimpleNamespace(
                                tensor_type=SimpleNamespace(
                                    shape=SimpleNamespace(
                                        dim=[
                                            SimpleNamespace(dim_value=1, dim_param=""),
                                            SimpleNamespace(
                                                dim_value=0, dim_param="height"
                                            ),
                                            SimpleNamespace(
                                                dim_value=0, dim_param="width"
                                            ),
                                        ]
                                    )
                                )
                            )
                        )
                    ]
                )
            )
        )
        monkeypatch.setitem(sys.modules, "onnx", dynamic_onnx)

        args = argparse.Namespace(
            debug=False,
            onnx_path=str(onnx_path),
            fp16=False,
            int8=False,
            output=None,
            config_path=None,
            calib_data=None,
            input_size=None,
            calib_samples=500,
            calib_batch_size=1,
            workspace_size=1 << 20,
        )

        pochi_cli.convert_command(args)

        assert FakeConverter.latest_instance is None
        error_messages = [
            str(call.args[0]) for call in logger.error.call_args_list if call.args
        ]
        assert any("動的シェイプ" in message for message in error_messages)

    def test_returns_when_tensorrt_unavailable(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """TensorRT利用不可時に変換処理へ進まないことを確認."""
        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_text("dummy", encoding="utf-8")

        logger = SimpleNamespace(
            debug=Mock(),
            info=Mock(),
            error=Mock(),
        )

        import pochitrain.tensorrt.converter as converter_module
        import pochitrain.tensorrt.inference as inference_module

        monkeypatch.setattr(converter_module, "TensorRTConverter", FakeConverter)
        monkeypatch.setattr(
            inference_module, "check_tensorrt_availability", lambda: False
        )
        monkeypatch.setattr(pochi_cli, "setup_logging", lambda debug=False: logger)

        args = argparse.Namespace(
            debug=False,
            onnx_path=str(onnx_path),
            fp16=False,
            int8=False,
            output=None,
            config_path=None,
            calib_data=None,
            input_size=[224, 224],
            calib_samples=500,
            calib_batch_size=1,
            workspace_size=1 << 20,
        )

        pochi_cli.convert_command(args)

        assert FakeConverter.latest_instance is None
        error_messages = [
            str(call.args[0]) for call in logger.error.call_args_list if call.args
        ]
        assert any("TensorRTが利用できません" in message for message in error_messages)

    def test_returns_when_onnx_path_does_not_exist(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """ONNXファイル未存在時に変換処理へ進まないことを確認."""
        onnx_path = tmp_path / "missing.onnx"

        logger = SimpleNamespace(
            debug=Mock(),
            info=Mock(),
            error=Mock(),
        )

        import pochitrain.tensorrt.converter as converter_module
        import pochitrain.tensorrt.inference as inference_module

        monkeypatch.setattr(converter_module, "TensorRTConverter", FakeConverter)
        monkeypatch.setattr(
            inference_module, "check_tensorrt_availability", lambda: True
        )
        monkeypatch.setattr(pochi_cli, "setup_logging", lambda debug=False: logger)

        args = argparse.Namespace(
            debug=False,
            onnx_path=str(onnx_path),
            fp16=False,
            int8=False,
            output=None,
            config_path=None,
            calib_data=None,
            input_size=[224, 224],
            calib_samples=500,
            calib_batch_size=1,
            workspace_size=1 << 20,
        )

        pochi_cli.convert_command(args)

        assert FakeConverter.latest_instance is None
        error_messages = [
            str(call.args[0]) for call in logger.error.call_args_list if call.args
        ]
        assert any(
            "ONNXモデルが見つかりません" in message for message in error_messages
        )


class TestPositiveIntValidation:
    """`positive_int` の境界値検証."""

    def test_positive_int_valid(self) -> None:
        """正の整数文字列は `int` に変換されることを確認."""
        assert positive_int("1") == 1
        assert positive_int("100") == 100

    def test_positive_int_invalid(self) -> None:
        """0以下は `ArgumentTypeError` になることを確認."""
        with pytest.raises(argparse.ArgumentTypeError):
            positive_int("0")
        with pytest.raises(argparse.ArgumentTypeError):
            positive_int("-1")
