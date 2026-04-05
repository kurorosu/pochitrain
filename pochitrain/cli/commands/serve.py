"""pochi serve サブコマンド."""

import argparse
from pathlib import Path
from typing import Any

import uvicorn

from pochitrain.api.app import create_app
from pochitrain.api.config import ServerConfig
from pochitrain.cli.cli_commons import setup_logging
from pochitrain.logging import COLORLOG_AVAILABLE, LOG_COLORS, LOG_DATE_FORMAT
from pochitrain.utils.inference_utils import validate_model_path

# uvicorn は %(module) だと全て "h11_impl" 等になるため %(name) を使用する.
# LoggerManager は %(module) (ファイル名) を使用している.
_UVICORN_COLOR_FORMAT = (
    "%(asctime)s|%(log_color)s%(levelname)-5.5s%(reset)s|"
    "%(name)-18s|%(lineno)03d| %(message)s"
)
_UVICORN_PLAIN_FORMAT = (
    "%(asctime)s|%(levelname)-5.5s|%(name)-18s|%(lineno)03d| %(message)s"
)


def _build_uvicorn_log_config(log_level: str) -> dict[str, Any]:
    """Uvicorn 用のログ設定を LoggerManager と同一フォーマットで生成する."""
    if COLORLOG_AVAILABLE:
        formatter_config: dict[str, Any] = {
            "()": "colorlog.ColoredFormatter",
            "format": _UVICORN_COLOR_FORMAT,
            "datefmt": LOG_DATE_FORMAT,
            "log_colors": LOG_COLORS,
        }
    else:
        formatter_config = {
            "format": _UVICORN_PLAIN_FORMAT,
            "datefmt": LOG_DATE_FORMAT,
        }

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": formatter_config,
            "access": formatter_config,
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn": {
                "handlers": ["default"],
                "level": log_level.upper(),
                "propagate": False,
            },
            "uvicorn.error": {
                "level": log_level.upper(),
                "handlers": ["default"],
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["access"],
                "level": log_level.upper(),
                "propagate": False,
            },
        },
    }


def serve_command(args: argparse.Namespace) -> None:
    """推論 API サーバーを起動する.

    Args:
        args: コマンドライン引数.
    """
    logger = setup_logging(debug=getattr(args, "debug", False))
    logger.info("=== pochitrain サーバーモード ===")

    model_path = Path(args.model_path)
    try:
        validate_model_path(model_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    server_config = ServerConfig(
        model_path=model_path,
        config_path=Path(args.config_path) if args.config_path else None,
        backend=args.backend,
        host=args.host,
        port=args.port,
    )

    app = create_app(server_config)

    logger.info(
        "推論 API サーバーを起動します: http://%s:%d",
        server_config.host,
        server_config.port,
    )

    debug = getattr(args, "debug", False)
    log_level = "debug" if debug else "info"
    uvicorn.run(
        app,
        host=server_config.host,
        port=server_config.port,
        log_level=log_level,
        log_config=_build_uvicorn_log_config(log_level),
    )
