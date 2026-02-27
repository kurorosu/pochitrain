"""pochi CLI train 系のテスト."""

import argparse
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torchvision.transforms as transforms

from pochitrain.cli.pochi import main, train_command


class TestMainDispatchTrain:
    """main の train ディスパッチを検証するテスト."""

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


class TestTrainCommandValidationHandling:
    """train_command の ValidationError ハンドリングのテスト."""

    @staticmethod
    def _build_minimal_config(**overrides: object) -> dict[str, object]:
        """train_command 用の最小設定辞書を返す."""
        config: dict[str, object] = {
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
        }
        config.update(overrides)
        return config

    @staticmethod
    def _make_train_args(tmp_path: Path) -> argparse.Namespace:
        """train_command 用の引数を生成する."""
        return argparse.Namespace(debug=False, config=str(tmp_path / "config.py"))

    @staticmethod
    def _patch_train_command_dependencies(
        monkeypatch: pytest.MonkeyPatch,
        module,
        logger: MagicMock,
        config: dict[str, object],
    ) -> MagicMock:
        """train_command の共通依存を差し替える."""
        monkeypatch.setattr(module, "setup_logging", lambda **_: logger)
        monkeypatch.setattr(module.signal, "signal", lambda *_: None)
        monkeypatch.setattr(
            module.ConfigLoader,
            "load_config",
            MagicMock(return_value=config),
        )
        create_data_loaders_mock = MagicMock()
        monkeypatch.setattr(module, "create_data_loaders", create_data_loaders_mock)
        return create_data_loaders_mock

    def test_train_command_validation_error_returns_early(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Pydantic 検証エラー時に早期 return することを確認する."""
        import pochitrain.cli.pochi as pochi_module

        logger = MagicMock()
        config = self._build_minimal_config(
            # ここだけ意図的に不正値を入れて ValidationError を発生させる.
            early_stopping={
                "enabled": False,
                "patience": 0,
            }
        )
        create_data_loaders_mock = self._patch_train_command_dependencies(
            monkeypatch,
            pochi_module,
            logger,
            config,
        )

        args = self._make_train_args(tmp_path)
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
        config = self._build_minimal_config(
            # train_data_root を存在しないパスへ差し替える.
            train_data_root=str(missing_train),
            val_data_root=str(existing_val),
        )
        create_data_loaders_mock = self._patch_train_command_dependencies(
            monkeypatch,
            pochi_module,
            logger,
            config,
        )

        args = self._make_train_args(tmp_path)
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
        config = self._build_minimal_config(
            # val_data_root を存在しないパスへ差し替える.
            train_data_root=str(existing_train),
            val_data_root=str(missing_val),
        )
        create_data_loaders_mock = self._patch_train_command_dependencies(
            monkeypatch,
            pochi_module,
            logger,
            config,
        )

        args = self._make_train_args(tmp_path)
        train_command(args)

        create_data_loaders_mock.assert_not_called()
        error_messages = [str(c.args[0]) for c in logger.error.call_args_list if c.args]
        assert any("検証データパスが存在しません" in msg for msg in error_messages)
