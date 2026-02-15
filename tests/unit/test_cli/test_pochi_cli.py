"""pochi CLIのテスト."""

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torchvision.transforms as transforms

from pochitrain.cli.pochi import (
    create_signal_handler,
    find_best_model,
    get_indexed_output_dir,
    infer_command,
    main,
    setup_logging,
    validate_config,
)


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

        # フラグを初期化
        pochi_module.training_interrupted = False

        # シグナルハンドラーを生成して呼び出し
        handler = create_signal_handler(debug=False)
        handler(2, None)

        # フラグが設定されていることを確認
        assert pochi_module.training_interrupted is True

        # クリーンアップ
        pochi_module.training_interrupted = False


class TestFindBestModel:
    """find_best_model関数のテスト."""

    def test_find_best_model_success(self, tmp_path):
        """ベストモデルを正しく検出することを確認."""
        # モデルディレクトリを作成
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        # ダミーモデルファイルを作成
        (models_dir / "best_epoch10.pth").touch()
        (models_dir / "best_epoch20.pth").touch()
        (models_dir / "best_epoch30.pth").touch()

        result = find_best_model(str(tmp_path))

        # 最大エポック番号のモデルが選択されることを確認
        assert result.name == "best_epoch30.pth"

    def test_find_best_model_cross_digit_boundary(self, tmp_path):
        """桁が変わるエポック番号でも正しく数値比較されることを確認."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        # best_epoch9.pth と best_epoch10.pth が共存するケース
        (models_dir / "best_epoch9.pth").touch()
        (models_dir / "best_epoch10.pth").touch()

        result = find_best_model(str(tmp_path))

        # 文字列比較だと "best_epoch9" > "best_epoch10" になるが,
        # 数値比較で epoch 10 が選択されることを確認
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

        # 連番が付与されることを確認
        assert result.name == "output_001"
        assert result.parent == tmp_path

    def test_get_indexed_output_dir_multiple(self, tmp_path):
        """複数の連番ディレクトリが存在する場合のテスト."""
        base_dir = tmp_path / "output"
        base_dir.mkdir()
        (tmp_path / "output_001").mkdir()
        (tmp_path / "output_002").mkdir()

        result = get_indexed_output_dir(str(base_dir))

        # 次の連番が選択されることを確認
        assert result.name == "output_003"


class TestValidateConfig:
    """validate_config関数のテスト."""

    def test_validate_config_valid(self, tmp_path):
        """有効な設定がTrueを返すことを確認."""
        # テスト用のデータディレクトリを作成
        train_dir = tmp_path / "train"
        val_dir = tmp_path / "val"
        train_dir.mkdir()
        val_dir.mkdir()

        # テスト用のtransformを作成
        test_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

        logger = setup_logging()
        config = {
            "model_name": "resnet18",
            "num_classes": 10,
            "pretrained": True,
            "train_data_root": str(train_dir),
            "val_data_root": str(val_dir),
            "batch_size": 32,
            "epochs": 10,
            "learning_rate": 0.001,
            "optimizer": "Adam",
            "device": "cpu",
            "work_dir": "work_dirs",
            "num_workers": 0,
            "train_transform": test_transform,
            "val_transform": test_transform,
            "enable_layer_wise_lr": False,
        }

        result = validate_config(config, logger)
        assert result is True

    def test_validate_config_invalid_optimizer(self, tmp_path):
        """無効なオプティマイザーがFalseを返すことを確認."""
        # テスト用のデータディレクトリを作成
        train_dir = tmp_path / "train"
        val_dir = tmp_path / "val"
        train_dir.mkdir()
        val_dir.mkdir()

        # テスト用のtransformを作成
        test_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

        logger = setup_logging()
        config = {
            "model_name": "resnet18",
            "num_classes": 10,
            "pretrained": True,
            "train_data_root": str(train_dir),
            "val_data_root": str(val_dir),
            "batch_size": 32,
            "epochs": 10,
            "learning_rate": 0.001,
            "optimizer": "InvalidOptimizer",
            "device": "cpu",
            "work_dir": "work_dirs",
            "num_workers": 0,
            "train_transform": test_transform,
            "val_transform": test_transform,
            "enable_layer_wise_lr": False,
        }

        result = validate_config(config, logger)
        assert result is False


class TestMainArgumentParsing:
    """main関数の引数パースのテスト."""

    def test_main_no_args_prints_help(self, monkeypatch: pytest.MonkeyPatch):
        """引数なしでヘルプを表示して終了することを確認."""
        monkeypatch.setattr("sys.argv", ["pochi"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    def test_main_dispatch_train_command(self, monkeypatch: pytest.MonkeyPatch):
        """trainサブコマンドのデフォルト設定を確認."""
        import pochitrain.cli.pochi as pochi_module

        called: dict[str, object] = {}

        def _fake_train(args: object) -> None:
            called["args"] = args

        monkeypatch.setattr("sys.argv", ["pochi", "train"])
        monkeypatch.setattr(pochi_module, "train_command", _fake_train)
        main()

        assert "args" in called
        assert getattr(called["args"], "command") == "train"

    def test_infer_parser_positional_model_path(self):
        """inferサブコマンドの位置引数model_pathを確認."""
        with patch("sys.argv", ["pochi", "infer", "model.pth"]):
            parser = argparse.ArgumentParser()
            parser.add_argument("--debug", action="store_true")
            subparsers = parser.add_subparsers(dest="command")
            infer_parser = subparsers.add_parser("infer")
            infer_parser.add_argument("model_path", help="モデルファイルパス")
            infer_parser.add_argument("--data", "-d")
            infer_parser.add_argument("--config-path", "-c")
            infer_parser.add_argument("--output", "-o")

            args = parser.parse_args(["infer", "model.pth"])
            assert args.model_path == "model.pth"
            assert args.data is None
            assert args.config_path is None
            assert args.output is None

    def test_infer_parser_with_optional_args(self):
        """inferサブコマンドのオプション引数を確認."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--debug", action="store_true")
        subparsers = parser.add_subparsers(dest="command")
        infer_parser = subparsers.add_parser("infer")
        infer_parser.add_argument("model_path", help="モデルファイルパス")
        infer_parser.add_argument("--data", "-d")
        infer_parser.add_argument("--config-path", "-c")
        infer_parser.add_argument("--output", "-o")

        args = parser.parse_args(
            ["infer", "model.pth", "-d", "data/val", "-c", "config.py"]
        )
        assert args.model_path == "model.pth"
        assert args.data == "data/val"
        assert args.config_path == "config.py"

    def test_optimize_parser_default_output(self):
        """optimizeサブコマンドのデフォルト出力ディレクトリを確認."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        optimize_parser = subparsers.add_parser("optimize")
        optimize_parser.add_argument(
            "--output",
            "-o",
            default="work_dirs/optuna_results",
        )

        args = parser.parse_args(["optimize"])
        assert args.output == "work_dirs/optuna_results"


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
    """infer_command が PyTorchInferenceService に正しく委譲するテスト."""

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

    @patch("pochitrain.cli.pochi.PyTorchInferenceService")
    @patch("pochitrain.cli.pochi.InferenceWorkspaceManager")
    @patch("pochitrain.cli.pochi.ConfigLoader")
    @patch("pochitrain.cli.pochi.validate_data_path")
    @patch("pochitrain.cli.pochi.validate_model_path")
    def test_delegates_to_service(
        self,
        mock_validate_model: MagicMock,
        mock_validate_data: MagicMock,
        mock_config_loader: MagicMock,
        mock_workspace_mgr_cls: MagicMock,
        mock_service_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Service の各メソッドが呼ばれることを検証する."""
        mock_config_loader.load_config.return_value = {
            "model_name": "resnet18",
            "num_classes": 2,
            "device": "cpu",
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 0.001,
            "optimizer": "Adam",
            "train_data_root": "data/train",
            "val_data_root": "data/val",
            "num_workers": 0,
            "train_transform": transforms.Compose(
                [transforms.Resize(224), transforms.ToTensor()]
            ),
            "val_transform": transforms.Compose(
                [transforms.Resize(224), transforms.ToTensor()]
            ),
            "enable_layer_wise_lr": False,
        }

        mock_service = mock_service_cls.return_value
        mock_service.create_predictor.return_value = MagicMock()
        mock_service.create_dataloader.return_value = (MagicMock(), MagicMock())
        mock_service.detect_input_size.return_value = (3, 224, 224)
        mock_service.run_inference.return_value = ([0, 1], [0.9, 0.8], {}, 100.0)

        mock_workspace = mock_workspace_mgr_cls.return_value
        mock_workspace.create_workspace.return_value = tmp_path / "workspace"
        mock_workspace.get_workspace_info.return_value = {
            "workspace_name": "test_workspace"
        }

        args = self._make_args(tmp_path)
        infer_command(args)

        mock_service.create_predictor.assert_called_once()
        mock_service.create_dataloader.assert_called_once()
        mock_service.detect_input_size.assert_called_once()
        mock_service.run_inference.assert_called_once()
        mock_service.aggregate_and_export.assert_called_once()

    @patch("pochitrain.cli.pochi.PyTorchInferenceService")
    @patch("pochitrain.cli.pochi.ConfigLoader")
    @patch("pochitrain.cli.pochi.validate_data_path")
    @patch("pochitrain.cli.pochi.validate_model_path")
    def test_predictor_error_returns_early(
        self,
        mock_validate_model: MagicMock,
        mock_validate_data: MagicMock,
        mock_config_loader: MagicMock,
        mock_service_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """推論器作成エラー時に早期 return すること."""
        mock_config_loader.load_config.return_value = {
            "model_name": "resnet18",
            "num_classes": 2,
            "device": "cpu",
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 0.001,
            "optimizer": "Adam",
            "train_data_root": "data/train",
            "val_data_root": "data/val",
            "num_workers": 0,
            "train_transform": transforms.Compose(
                [transforms.Resize(224), transforms.ToTensor()]
            ),
            "val_transform": transforms.Compose(
                [transforms.Resize(224), transforms.ToTensor()]
            ),
            "enable_layer_wise_lr": False,
        }

        mock_service = mock_service_cls.return_value
        mock_service.create_predictor.side_effect = RuntimeError("model error")

        args = self._make_args(tmp_path)
        infer_command(args)

        mock_service.create_predictor.assert_called_once()
        mock_service.create_dataloader.assert_not_called()

    @patch("pochitrain.cli.pochi.PyTorchInferenceService")
    @patch("pochitrain.cli.pochi.ConfigLoader")
    @patch("pochitrain.cli.pochi.validate_data_path")
    @patch("pochitrain.cli.pochi.validate_model_path")
    def test_inference_error_returns_early(
        self,
        mock_validate_model: MagicMock,
        mock_validate_data: MagicMock,
        mock_config_loader: MagicMock,
        mock_service_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """推論実行エラー時に早期 return すること."""
        mock_config_loader.load_config.return_value = {
            "model_name": "resnet18",
            "num_classes": 2,
            "device": "cpu",
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 0.001,
            "optimizer": "Adam",
            "train_data_root": "data/train",
            "val_data_root": "data/val",
            "num_workers": 0,
            "train_transform": transforms.Compose(
                [transforms.Resize(224), transforms.ToTensor()]
            ),
            "val_transform": transforms.Compose(
                [transforms.Resize(224), transforms.ToTensor()]
            ),
            "enable_layer_wise_lr": False,
        }

        mock_service = mock_service_cls.return_value
        mock_service.create_predictor.return_value = MagicMock()
        mock_service.create_dataloader.return_value = (MagicMock(), MagicMock())
        mock_service.detect_input_size.return_value = None
        mock_service.run_inference.side_effect = RuntimeError("inference error")

        args = self._make_args(tmp_path)
        infer_command(args)

        mock_service.run_inference.assert_called_once()
        mock_service.aggregate_and_export.assert_not_called()
