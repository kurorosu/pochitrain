"""pochi CLI共通機能のテスト."""

from pathlib import Path

import pytest

from pochitrain.cli.pochi import (
    create_signal_handler,
    find_best_model,
    get_indexed_output_dir,
    main,
    setup_logging,
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
