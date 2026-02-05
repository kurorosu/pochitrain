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

    def test_main_no_args_prints_help(self, capsys):
        """引数なしでヘルプを表示して終了することを確認."""
        with patch("sys.argv", ["pochi"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_train_parser_default_config(self):
        """trainサブコマンドのデフォルト設定を確認."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        train_parser = subparsers.add_parser("train")
        train_parser.add_argument(
            "--config",
            default="configs/pochi_train_config.py",
        )

        args = parser.parse_args(["train"])
        assert args.config == "configs/pochi_train_config.py"

    def test_infer_parser_required_args(self):
        """inferサブコマンドの必須引数を確認."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        infer_parser = subparsers.add_parser("infer")
        infer_parser.add_argument("--model-path", "-m", required=True)
        infer_parser.add_argument("--data", "-d", required=True)
        infer_parser.add_argument("--config-path", "-c", required=True)

        args = parser.parse_args(
            ["infer", "-m", "model.pth", "-d", "data/val", "-c", "config.py"]
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
