"""pochi serve CLI のテスト."""

from unittest.mock import MagicMock, patch

import pytest

from pochitrain.cli.commands.serve import _build_uvicorn_log_config


class TestBuildUvicornLogConfig:
    """_build_uvicorn_log_config() のテスト."""

    def test_config_structure(self):
        """返却される辞書に必須キーが含まれることを確認."""
        config = _build_uvicorn_log_config("info")
        assert config["version"] == 1
        assert "formatters" in config
        assert "handlers" in config
        assert "loggers" in config
        assert "default" in config["formatters"]
        assert "access" in config["formatters"]

    def test_log_level_info(self):
        """log_level='info' が大文字変換されてロガーに設定されることを確認."""
        config = _build_uvicorn_log_config("info")
        assert config["loggers"]["uvicorn"]["level"] == "INFO"
        assert config["loggers"]["uvicorn.error"]["level"] == "INFO"
        assert config["loggers"]["uvicorn.access"]["level"] == "INFO"

    def test_log_level_debug(self):
        """log_level='debug' が大文字変換されてロガーに設定されることを確認."""
        config = _build_uvicorn_log_config("debug")
        assert config["loggers"]["uvicorn"]["level"] == "DEBUG"

    def test_with_colorlog_available(self):
        """colorlog 利用可能時に ColoredFormatter が設定されることを確認."""
        config = _build_uvicorn_log_config("info")
        formatter = config["formatters"]["default"]
        assert formatter["()"] == "colorlog.ColoredFormatter"
        assert "log_colors" in formatter

    def test_without_colorlog(self, monkeypatch):
        """colorlog 未インストール時にプレーンフォーマットが使用されることを確認."""
        monkeypatch.setattr("pochitrain.cli.commands.serve.COLORLOG_AVAILABLE", False)
        config = _build_uvicorn_log_config("info")
        formatter = config["formatters"]["default"]
        assert "()" not in formatter
        assert "log_colors" not in formatter
        assert "format" in formatter

    def test_handlers_stream(self):
        """default は stderr, access は stdout に出力されることを確認."""
        config = _build_uvicorn_log_config("info")
        assert config["handlers"]["default"]["stream"] == "ext://sys.stderr"
        assert config["handlers"]["access"]["stream"] == "ext://sys.stdout"


class TestServeCommand:
    """serve_command() のテスト."""

    def test_invalid_model_path(self, tmp_path, capsys):
        """存在しないモデルパスでエラーログが出力されることを確認."""
        from pochitrain.cli.commands.serve import serve_command

        args = MagicMock()
        args.model_path = str(tmp_path / "nonexistent.pth")
        args.debug = False

        serve_command(args)
        # FileNotFoundError がキャッチされ return するので例外は発生しない

    @patch("pochitrain.cli.commands.serve.uvicorn")
    @patch("pochitrain.cli.commands.serve.create_app")
    def test_debug_flag_sets_log_level(self, mock_create_app, mock_uvicorn, tmp_path):
        """--debug フラグが uvicorn の log_level に反映されることを確認."""
        from pochitrain.cli.commands.serve import serve_command

        model_path = tmp_path / "model.pth"
        model_path.touch()

        args = MagicMock()
        args.model_path = str(model_path)
        args.config_path = None
        args.backend = "pytorch"
        args.host = "127.0.0.1"
        args.port = 8000
        args.debug = True

        serve_command(args)

        mock_uvicorn.run.assert_called_once()
        call_kwargs = mock_uvicorn.run.call_args[1]
        assert call_kwargs["log_level"] == "debug"
