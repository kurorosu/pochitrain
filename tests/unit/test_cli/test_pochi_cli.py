"""pochi CLIのテスト."""

import argparse
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torchvision.transforms as transforms

from pochitrain.cli.pochi import (
    create_signal_handler,
    find_best_model,
    get_indexed_output_dir,
    infer_command,
    main,
    setup_logging,
    train_command,
)
from pochitrain.inference.types.orchestration_types import InferenceRunResult


class TestSetupLogging:
    """setup_logging関数のテスト."""

    def test_setup_logging_returns_logger(self):
        """ロガーを返すことを確認."""
        logger = setup_logging()
        assert logger is not None
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")

    def test_setup_logging_with_custom_name(self):
        """カスタム名でロガーを作成できることを確認."""
        logger = setup_logging("custom_logger")
        assert logger is not None


class TestSignalHandler:
    """signal_handler関数のテスト."""

    def test_signal_handler_sets_flag(self):
        """シグナルハンドラーが停止フラグを設定することを確認."""
        import pochitrain.cli.pochi as pochi_module

        pochi_module.training_interrupted = False

        handler = create_signal_handler(debug=False)
        handler(2, None)

        assert pochi_module.training_interrupted is True

        pochi_module.training_interrupted = False


class TestFindBestModel:
    """find_best_model関数のテスト."""

    def test_find_best_model_success(self, tmp_path):
        """ベストモデルを正しく検出することを確認."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        (models_dir / "best_epoch10.pth").touch()
        (models_dir / "best_epoch20.pth").touch()
        (models_dir / "best_epoch30.pth").touch()

        result = find_best_model(str(tmp_path))

        assert result.name == "best_epoch30.pth"

    def test_find_best_model_cross_digit_boundary(self, tmp_path):
        """桁が変わるエポック番号でも正しく数値比較されることを確認."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        (models_dir / "best_epoch9.pth").touch()
        (models_dir / "best_epoch10.pth").touch()

        result = find_best_model(str(tmp_path))

        # 文字列比較では 9 が 10 より大きく見えるため, 数値比較を検証する.
        assert result.name == "best_epoch10.pth"

    def test_find_best_model_no_models_dir(self, tmp_path):
        """モデルディレクトリがない場合にエラーを発生させることを確認."""
        with pytest.raises(
            FileNotFoundError, match="モデルディレクトリが見つかりません"
        ):
            find_best_model(str(tmp_path))

    def test_find_best_model_no_model_files(self, tmp_path):
        """モデルファイルがない場合にエラーを発生させることを確認."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="ベストモデルが見つかりません"):
            find_best_model(str(tmp_path))


class TestGetIndexedOutputDir:
    """get_indexed_output_dir関数のテスト."""

    def test_get_indexed_output_dir_new(self, tmp_path):
        """存在しないディレクトリはそのまま返すことを確認."""
        new_dir = tmp_path / "new_output"
        result = get_indexed_output_dir(str(new_dir))
        assert result == new_dir

    def test_get_indexed_output_dir_existing(self, tmp_path):
        """存在するディレクトリは連番を付与することを確認."""
        existing_dir = tmp_path / "output"
        existing_dir.mkdir()

        result = get_indexed_output_dir(str(existing_dir))

        assert result.name == "output_001"
        assert result.parent == tmp_path

    def test_get_indexed_output_dir_multiple(self, tmp_path):
        """複数の連番ディレクトリが存在する場合のテスト."""
        base_dir = tmp_path / "output"
        base_dir.mkdir()
        (tmp_path / "output_001").mkdir()
        (tmp_path / "output_002").mkdir()

        result = get_indexed_output_dir(str(base_dir))

        assert result.name == "output_003"


class TestMainArgumentParsing:
    """main関数の引数パースのテスト."""

    def test_main_no_args_prints_help(self, monkeypatch: pytest.MonkeyPatch):
        """引数なしでヘルプを表示して終了することを確認."""
        monkeypatch.setattr("sys.argv", ["pochi"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1


class TestMainDispatch:
    """main のディスパッチ経路を検証するテスト."""

    def test_dispatch_train_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """train サブコマンドで train_command が呼ばれることを検証する."""
        import pochitrain.cli.pochi as pochi_module

        called: dict[str, object] = {}

        def _fake_train(args: object) -> None:
            called["args"] = args

        monkeypatch.setattr("sys.argv", ["pochi", "train"])
        monkeypatch.setattr(pochi_module, "train_command", _fake_train)
        main()

        assert "args" in called
        assert getattr(called["args"], "command") == "train"

    def test_dispatch_infer_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """infer サブコマンドで infer_command が呼ばれることを検証する."""
        import pochitrain.cli.pochi as pochi_module

        called: dict[str, object] = {}

        def _fake_infer(args: object) -> None:
            called["args"] = args

        monkeypatch.setattr(
            "sys.argv",
            [
                "pochi",
                "infer",
                "model.pth",
                "--data",
                "data/val",
                "--config-path",
                "config.py",
            ],
        )
        monkeypatch.setattr(pochi_module, "infer_command", _fake_infer)
        main()

        assert "args" in called
        assert getattr(called["args"], "command") == "infer"
        assert getattr(called["args"], "model_path") == "model.pth"
        assert getattr(called["args"], "data") == "data/val"
        assert getattr(called["args"], "config_path") == "config.py"

    def test_dispatch_optimize_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """optimize サブコマンドで optimize_command が呼ばれることを検証する."""
        import pochitrain.cli.pochi as pochi_module

        called: dict[str, object] = {}

        def _fake_optimize(args: object) -> None:
            called["args"] = args

        monkeypatch.setattr("sys.argv", ["pochi", "optimize"])
        monkeypatch.setattr(pochi_module, "optimize_command", _fake_optimize)
        main()

        assert "args" in called
        assert getattr(called["args"], "command") == "optimize"
        assert getattr(called["args"], "output") == "work_dirs/optuna_results"

    def test_dispatch_convert_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """convert サブコマンドで convert_command が呼ばれることを検証する."""
        import pochitrain.cli.pochi as pochi_module

        called: dict[str, object] = {}

        def _fake_convert(args: object) -> None:
            called["args"] = args

        monkeypatch.setattr("sys.argv", ["pochi", "convert", "model.onnx"])
        monkeypatch.setattr(pochi_module, "convert_command", _fake_convert)
        main()

        assert "args" in called
        assert getattr(called["args"], "command") == "convert"
        assert getattr(called["args"], "onnx_path") == "model.onnx"


class TestInferCommandServiceDelegation:
    """infer_command が Service に委譲するテスト."""

    class _Dataset:
        labels = [0, 1]

        @staticmethod
        def get_file_paths() -> list[str]:
            return ["a.jpg", "b.jpg"]

        @staticmethod
        def get_classes() -> list[str]:
            return ["cat", "dog"]

    class _Predictor:
        @staticmethod
        def get_model_info() -> dict[str, str]:
            return {"model_name": "resnet18"}

    class _ServiceStub:
        def __init__(self, resolved_output_dir: Path) -> None:
            self.resolved_output_dir = resolved_output_dir
            self.aggregate_called = False
            self.create_dataloader_called = False
            self.raise_on_create_predictor = False
            self.raise_on_run = False
            self.captured: dict[str, object] = {}

        def resolve_paths(self, request, config):
            del config
            data_path = request.data_path if request.data_path is not None else Path("")
            output_dir = (
                request.output_dir
                if request.output_dir is not None
                else self.resolved_output_dir
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            return SimpleNamespace(
                model_path=request.model_path,
                data_path=data_path,
                output_dir=output_dir,
            )

        def resolve_pipeline(self, requested: str, *, use_gpu: bool) -> str:
            del use_gpu
            return requested

        def resolve_runtime_options(self, *, config, pipeline: str, use_gpu: bool):
            return SimpleNamespace(
                pipeline=pipeline,
                batch_size=1,
                num_workers=0,
                pin_memory=bool(config.get("infer_pin_memory", True)),
                use_gpu=use_gpu,
                use_gpu_pipeline=pipeline == "gpu",
            )

        def create_predictor(self, config, model_path):
            del config
            del model_path
            if self.raise_on_create_predictor:
                raise RuntimeError("model error")
            return TestInferCommandServiceDelegation._Predictor()

        def create_dataloader(
            self,
            config,
            data_path,
            val_transform,
            pipeline,
            runtime_options,
        ):
            del config
            del val_transform
            self.create_dataloader_called = True
            self.captured["data_path"] = data_path
            self.captured["pipeline"] = pipeline
            self.captured["pin_memory"] = runtime_options.pin_memory
            return (
                object(),
                TestInferCommandServiceDelegation._Dataset(),
                pipeline,
                None,
                None,
            )

        def detect_input_size(self, config, dataset):
            del config
            del dataset
            return (3, 224, 224)

        def create_runtime_adapter(self, predictor):
            del predictor
            return SimpleNamespace(use_cuda_timing=False)

        def build_runtime_execution_request(self, **kwargs):
            self.captured["use_gpu_pipeline"] = kwargs["use_gpu_pipeline"]
            self.captured["gpu_non_blocking"] = kwargs["gpu_non_blocking"]
            return SimpleNamespace(
                execution_request=SimpleNamespace(
                    gpu_non_blocking=kwargs["gpu_non_blocking"]
                )
            )

        def run(self, runtime_request):
            del runtime_request
            if self.raise_on_run:
                raise RuntimeError("inference error")
            return InferenceRunResult(
                predictions=[0, 1],
                confidences=[0.9, 0.8],
                true_labels=[0, 1],
                num_samples=2,
                correct=2,
                avg_time_per_image=1.0,
                total_samples=2,
                warmup_samples=0,
                avg_total_time_per_image=1.5,
            )

        def aggregate_and_export(self, **kwargs):
            self.aggregate_called = True
            self.captured["workspace_dir"] = kwargs["workspace_dir"]

    def _make_args(self, tmp_path: Path) -> argparse.Namespace:
        """テスト用の argparse.Namespace を生成する."""
        model_path = tmp_path / "models" / "best.pth"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.touch()

        data_path = tmp_path / "data"
        data_path.mkdir(exist_ok=True)

        config_path = tmp_path / "config.py"

        return argparse.Namespace(
            debug=False,
            model_path=str(model_path),
            data=str(data_path),
            config_path=str(config_path),
            output=str(tmp_path / "output"),
        )

    @staticmethod
    def _install_service_stub(
        monkeypatch: pytest.MonkeyPatch,
        module,
        service_stub: "_ServiceStub",
    ) -> None:
        monkeypatch.setattr(
            module,
            "PyTorchInferenceService",
            lambda logger: service_stub,
        )

    @staticmethod
    def _build_config_dict(
        data_root: Path, *, infer_pin_memory: bool = True
    ) -> dict[str, object]:
        """infer_command 用の最小設定辞書を生成する."""
        return {
            "model_name": "resnet18",
            "num_classes": 2,
            "device": "cpu",
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 0.001,
            "optimizer": "Adam",
            "train_data_root": "data/train",
            "val_data_root": str(data_root),
            "num_workers": 0,
            "infer_pin_memory": infer_pin_memory,
            "train_transform": transforms.Compose(
                [transforms.Resize(224), transforms.ToTensor()]
            ),
            "val_transform": transforms.Compose(
                [transforms.Resize(224), transforms.ToTensor()]
            ),
            "enable_layer_wise_lr": False,
        }

    def test_delegates_to_service_with_explicit_output(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """--output 指定時に Service 解決結果を使って委譲することを検証する."""
        import pochitrain.cli.pochi as pochi_module

        args = self._make_args(tmp_path)
        config = self._build_config_dict(Path(args.data), infer_pin_memory=False)
        monkeypatch.setattr(
            pochi_module.ConfigLoader,
            "load_config",
            lambda _path: config,
        )
        service_stub = self._ServiceStub(resolved_output_dir=tmp_path / "unused")
        self._install_service_stub(monkeypatch, pochi_module, service_stub)

        infer_command(args)

        assert service_stub.captured["data_path"] == Path(args.data)
        assert service_stub.captured["pipeline"] == "current"
        assert service_stub.captured["pin_memory"] is False
        assert service_stub.captured["workspace_dir"] == Path(args.output)
        assert Path(args.output).exists()

    def test_creates_workspace_when_output_not_specified(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """--output 未指定時は model 位置基準で workspace が生成されることを検証する."""
        import pochitrain.cli.pochi as pochi_module

        work_dir = tmp_path / "work_dirs" / "20260223_001"
        model_path = work_dir / "models" / "best.pth"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.touch()

        data_path = tmp_path / "data"
        data_path.mkdir(parents=True, exist_ok=True)

        args = argparse.Namespace(
            debug=False,
            model_path=str(model_path),
            data=str(data_path),
            config_path=str(tmp_path / "config.py"),
            output=None,
        )
        config = self._build_config_dict(data_path)
        monkeypatch.setattr(
            pochi_module.ConfigLoader,
            "load_config",
            lambda _path: config,
        )
        expected_output = work_dir / "inference_results" / "20260223_001"
        service_stub = self._ServiceStub(resolved_output_dir=expected_output)
        self._install_service_stub(monkeypatch, pochi_module, service_stub)

        infer_command(args)

        workspace_dir_obj = service_stub.captured["workspace_dir"]
        assert isinstance(workspace_dir_obj, Path)
        workspace_dir = workspace_dir_obj
        assert workspace_dir.parent == work_dir / "inference_results"
        assert workspace_dir.exists()

    def test_predictor_error_returns_early(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """推論器作成エラー時に早期 return することを検証する."""
        import pochitrain.cli.pochi as pochi_module

        args = self._make_args(tmp_path)
        config = self._build_config_dict(Path(args.data))
        monkeypatch.setattr(
            pochi_module.ConfigLoader,
            "load_config",
            lambda _path: config,
        )
        service_stub = self._ServiceStub(resolved_output_dir=tmp_path / "unused")
        service_stub.raise_on_create_predictor = True
        self._install_service_stub(monkeypatch, pochi_module, service_stub)

        infer_command(args)

        assert service_stub.create_dataloader_called is False
        assert service_stub.aggregate_called is False

    def test_inference_error_returns_early(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """推論実行エラー時に早期 return することを検証する."""
        import pochitrain.cli.pochi as pochi_module

        args = self._make_args(tmp_path)
        config = self._build_config_dict(Path(args.data))
        monkeypatch.setattr(
            pochi_module.ConfigLoader,
            "load_config",
            lambda _path: config,
        )
        service_stub = self._ServiceStub(resolved_output_dir=tmp_path / "unused")
        service_stub.raise_on_run = True
        self._install_service_stub(monkeypatch, pochi_module, service_stub)

        infer_command(args)

        assert service_stub.aggregate_called is False


class TestTrainCommandValidationHandling:
    """train_command の ValidationError ハンドリングのテスト."""

    def test_train_command_validation_error_returns_early(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Pydantic 検証エラー時に早期 return することを確認する."""
        import pochitrain.cli.pochi as pochi_module

        logger = MagicMock()
        config = {
            "model_name": "resnet18",
            "num_classes": 2,
            "device": "cpu",
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 0.001,
            "optimizer": "Adam",
            "train_data_root": "data/train",
            "val_data_root": "data/val",
            "train_transform": transforms.Compose([transforms.ToTensor()]),
            "val_transform": transforms.Compose([transforms.ToTensor()]),
            "enable_layer_wise_lr": False,
            "early_stopping": {
                "enabled": False,
                "patience": 0,
            },
        }

        monkeypatch.setattr(pochi_module, "setup_logging", lambda **_: logger)
        monkeypatch.setattr(pochi_module.signal, "signal", lambda *_: None)
        monkeypatch.setattr(
            pochi_module.ConfigLoader,
            "load_config",
            MagicMock(return_value=config),
        )
        create_data_loaders_mock = MagicMock()
        monkeypatch.setattr(
            pochi_module, "create_data_loaders", create_data_loaders_mock
        )

        args = argparse.Namespace(debug=False, config=str(tmp_path / "config.py"))
        train_command(args)

        create_data_loaders_mock.assert_not_called()
        error_messages = [str(c.args[0]) for c in logger.error.call_args_list if c.args]
        assert any("設定にエラーがあります" in msg for msg in error_messages)

    def test_train_command_returns_early_when_train_data_root_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """train_data_root が存在しない場合に早期 return することを確認する."""
        import pochitrain.cli.pochi as pochi_module

        logger = MagicMock()
        missing_train = tmp_path / "missing_train"
        existing_val = tmp_path / "val"
        existing_val.mkdir()

        config = {
            "model_name": "resnet18",
            "num_classes": 2,
            "device": "cpu",
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 0.001,
            "optimizer": "Adam",
            "train_data_root": str(missing_train),
            "val_data_root": str(existing_val),
            "train_transform": transforms.Compose([transforms.ToTensor()]),
            "val_transform": transforms.Compose([transforms.ToTensor()]),
            "enable_layer_wise_lr": False,
        }

        monkeypatch.setattr(pochi_module, "setup_logging", lambda **_: logger)
        monkeypatch.setattr(pochi_module.signal, "signal", lambda *_: None)
        monkeypatch.setattr(
            pochi_module.ConfigLoader,
            "load_config",
            MagicMock(return_value=config),
        )
        create_data_loaders_mock = MagicMock()
        monkeypatch.setattr(
            pochi_module, "create_data_loaders", create_data_loaders_mock
        )

        args = argparse.Namespace(debug=False, config=str(tmp_path / "config.py"))
        train_command(args)

        create_data_loaders_mock.assert_not_called()
        error_messages = [str(c.args[0]) for c in logger.error.call_args_list if c.args]
        assert any("訓練データパスが存在しません" in msg for msg in error_messages)

    def test_train_command_returns_early_when_val_data_root_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """val_data_root が存在しない場合に早期 return することを確認する."""
        import pochitrain.cli.pochi as pochi_module

        logger = MagicMock()
        existing_train = tmp_path / "train"
        existing_train.mkdir()
        missing_val = tmp_path / "missing_val"

        config = {
            "model_name": "resnet18",
            "num_classes": 2,
            "device": "cpu",
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 0.001,
            "optimizer": "Adam",
            "train_data_root": str(existing_train),
            "val_data_root": str(missing_val),
            "train_transform": transforms.Compose([transforms.ToTensor()]),
            "val_transform": transforms.Compose([transforms.ToTensor()]),
            "enable_layer_wise_lr": False,
        }

        monkeypatch.setattr(pochi_module, "setup_logging", lambda **_: logger)
        monkeypatch.setattr(pochi_module.signal, "signal", lambda *_: None)
        monkeypatch.setattr(
            pochi_module.ConfigLoader,
            "load_config",
            MagicMock(return_value=config),
        )
        create_data_loaders_mock = MagicMock()
        monkeypatch.setattr(
            pochi_module, "create_data_loaders", create_data_loaders_mock
        )

        args = argparse.Namespace(debug=False, config=str(tmp_path / "config.py"))
        train_command(args)

        create_data_loaders_mock.assert_not_called()
        error_messages = [str(c.args[0]) for c in logger.error.call_args_list if c.args]
        assert any("検証データパスが存在しません" in msg for msg in error_messages)
