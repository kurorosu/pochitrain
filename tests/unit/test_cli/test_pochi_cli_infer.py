"""pochi CLI infer 系のテスト."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from pochitrain.cli.pochi import infer_command, main
from pochitrain.inference.types.orchestration_types import InferenceRunResult

from .conftest import build_cli_config


class TestMainDispatchInfer:
    """main の infer ディスパッチを検証するテスト."""

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


class _Dataset:
    """テスト用の最小データセットスタブ."""

    labels = [0, 1]

    @staticmethod
    def get_file_paths() -> list[str]:
        """ファイルパス一覧を返す."""
        return ["a.jpg", "b.jpg"]

    @staticmethod
    def get_classes() -> list[str]:
        """クラス一覧を返す."""
        return ["cat", "dog"]


class _Predictor:
    """テスト用の最小推論器スタブ."""

    @staticmethod
    def get_model_info() -> dict[str, str]:
        """モデル情報を返す."""
        return {"model_name": "resnet18"}


class _ServiceStub:
    """PyTorchInferenceService の最小スタブ.

    観測可能な副作用 (aggregate_called, create_dataloader_called) のみ記録し,
    内部引数の検証は行わない.
    """

    def __init__(self, resolved_output_dir: Path) -> None:
        self.resolved_output_dir = resolved_output_dir
        self.aggregate_called = False
        self.create_dataloader_called = False
        self.raise_on_create_predictor = False
        self.raise_on_run = False

    def resolve_paths(self, request, config):
        """パス解決をスタブする."""
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

    def resolve_pipeline(self, requested, *, use_gpu):
        """パイプライン解決をスタブする."""
        del use_gpu
        return requested

    def resolve_runtime_options(self, *, config, pipeline, use_gpu):
        """ランタイムオプション解決をスタブする."""
        return SimpleNamespace(
            pipeline=pipeline,
            batch_size=1,
            num_workers=0,
            pin_memory=bool(config.get("infer_pin_memory", True)),
            use_gpu=use_gpu,
            use_gpu_pipeline=pipeline == "gpu",
        )

    def create_predictor(self, config, model_path):
        """推論器作成をスタブする."""
        del config, model_path
        if self.raise_on_create_predictor:
            raise RuntimeError("model error")
        return _Predictor()

    def create_dataloader(
        self, config, data_path, val_transform, pipeline, runtime_options
    ):
        """データローダー作成をスタブする."""
        del config, val_transform, runtime_options
        self.create_dataloader_called = True
        return (object(), _Dataset(), pipeline, None, None)

    def detect_input_size(self, config, dataset):
        """入力サイズ検出をスタブする."""
        del config, dataset
        return (3, 224, 224)

    def create_runtime_adapter(self, predictor):
        """ランタイムアダプター作成をスタブする."""
        del predictor
        return SimpleNamespace(use_cuda_timing=False)

    def build_runtime_execution_request(self, **kwargs):
        """実行リクエスト構築をスタブする."""
        return SimpleNamespace(
            execution_request=SimpleNamespace(
                gpu_non_blocking=kwargs.get("gpu_non_blocking", True),
            ),
        )

    def run(self, runtime_request):
        """推論実行をスタブする."""
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
        """集計・出力をスタブする."""
        del kwargs
        self.aggregate_called = True


def _create_dummy_model_file(base_dir: Path) -> Path:
    """ダミーモデルファイルを作成して返す."""
    model_path = base_dir / "models" / "best.pth"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.touch()
    return model_path


def _create_dummy_data_dir(base_dir: Path) -> Path:
    """ダミー推論データディレクトリを作成して返す."""
    data_path = base_dir / "data"
    data_path.mkdir(parents=True, exist_ok=True)
    return data_path


def _make_args(tmp_path: Path) -> argparse.Namespace:
    """テスト用の argparse.Namespace を生成する."""
    model_path = _create_dummy_model_file(tmp_path)
    data_path = _create_dummy_data_dir(tmp_path)
    return argparse.Namespace(
        debug=False,
        model_path=str(model_path),
        data=str(data_path),
        config_path=str(tmp_path / "config.py"),
        output=str(tmp_path / "output"),
    )


def _install_service_stub(
    monkeypatch: pytest.MonkeyPatch,
    module: object,
    service_stub: _ServiceStub,
) -> None:
    """ServiceStub を PyTorchInferenceService の代わりに注入する."""
    monkeypatch.setattr(
        module,
        "PyTorchInferenceService",
        lambda logger: service_stub,
    )


class TestInferCommandServiceDelegation:
    """infer_command が Service に委譲するテスト."""

    def test_successful_inference_creates_output_dir(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """正常完了時に出力ディレクトリが作成され集計が実行されることを検証する."""
        import pochitrain.cli.pochi as pochi_module

        args = _make_args(tmp_path)
        config = build_cli_config(val_data_root=args.data)
        monkeypatch.setattr(
            pochi_module.ConfigLoader,
            "load_config",
            lambda _path: config,
        )
        service_stub = _ServiceStub(resolved_output_dir=tmp_path / "unused")
        _install_service_stub(monkeypatch, pochi_module, service_stub)

        infer_command(args)

        assert Path(args.output).exists()
        assert service_stub.aggregate_called is True

    def test_creates_workspace_when_output_not_specified(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """--output 未指定時は model 位置基準で workspace が生成されることを検証する."""
        import pochitrain.cli.pochi as pochi_module

        work_dir = tmp_path / "work_dirs" / "20260223_001"
        model_path = _create_dummy_model_file(work_dir)
        data_path = _create_dummy_data_dir(tmp_path)

        args = argparse.Namespace(
            debug=False,
            model_path=str(model_path),
            data=str(data_path),
            config_path=str(tmp_path / "config.py"),
            output=None,
        )
        config = build_cli_config(val_data_root=str(data_path))
        monkeypatch.setattr(
            pochi_module.ConfigLoader,
            "load_config",
            lambda _path: config,
        )
        expected_output = work_dir / "inference_results" / "20260223_001"
        service_stub = _ServiceStub(resolved_output_dir=expected_output)
        _install_service_stub(monkeypatch, pochi_module, service_stub)

        infer_command(args)

        assert expected_output.exists()
        assert service_stub.aggregate_called is True

    def test_predictor_error_skips_dataloader_and_export(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """推論器作成エラー時にデータローダー作成と集計がスキップされることを検証する."""
        import pochitrain.cli.pochi as pochi_module

        args = _make_args(tmp_path)
        config = build_cli_config(val_data_root=args.data)
        monkeypatch.setattr(
            pochi_module.ConfigLoader,
            "load_config",
            lambda _path: config,
        )
        service_stub = _ServiceStub(resolved_output_dir=tmp_path / "unused")
        service_stub.raise_on_create_predictor = True
        _install_service_stub(monkeypatch, pochi_module, service_stub)

        infer_command(args)

        assert service_stub.create_dataloader_called is False
        assert service_stub.aggregate_called is False

    def test_inference_error_skips_export(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """推論実行エラー時に集計がスキップされることを検証する."""
        import pochitrain.cli.pochi as pochi_module

        args = _make_args(tmp_path)
        config = build_cli_config(val_data_root=args.data)
        monkeypatch.setattr(
            pochi_module.ConfigLoader,
            "load_config",
            lambda _path: config,
        )
        service_stub = _ServiceStub(resolved_output_dir=tmp_path / "unused")
        service_stub.raise_on_run = True
        _install_service_stub(monkeypatch, pochi_module, service_stub)

        infer_command(args)

        assert service_stub.aggregate_called is False
