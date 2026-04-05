"""pochi serve サブコマンド."""

import argparse
from pathlib import Path

import uvicorn

from pochitrain.api.app import create_app
from pochitrain.api.config import ServerConfig
from pochitrain.cli.cli_commons import setup_logging


def serve_command(args: argparse.Namespace) -> None:
    """推論 API サーバーを起動する.

    Args:
        args: コマンドライン引数.
    """
    logger = setup_logging(debug=getattr(args, "debug", False))
    logger.info("=== pochitrain サーバーモード ===")

    server_config = ServerConfig(
        model_path=Path(args.model_path),
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

    uvicorn.run(
        app,
        host=server_config.host,
        port=server_config.port,
        log_level="info",
    )
