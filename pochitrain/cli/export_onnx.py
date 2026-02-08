#!/usr/bin/env python3
"""PyTorchモデル(.pth)をONNX形式に変換するCLI.

使用例:
    uv run export-onnx work_dirs/20251018_001/models/best_epoch40.pth --input-size 224 224
    uv run export-onnx model.pth --config work_dirs/20251018_001/config.py
    uv run export-onnx model.pth --model-name resnet18 --num-classes 4 --input-size 224 224
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

from pochitrain.cli.arg_types import positive_int
from pochitrain.logging import LoggerManager
from pochitrain.onnx import OnnxExporter
from pochitrain.utils import ConfigLoader

logger: logging.Logger = LoggerManager().get_logger(__name__)


def main() -> None:
    """メイン関数."""
    parser = argparse.ArgumentParser(
        description="PyTorchモデル(.pth)をONNX形式に変換",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 設定ファイルから情報を取得（推奨）
  uv run export-onnx work_dirs/20251018_001/models/best_epoch40.pth --input-size 224 224

  # 設定ファイルを明示的に指定
  uv run export-onnx model.pth --config work_dirs/20251018_001/config.py --input-size 224 224

  # モデル情報を直接指定
  uv run export-onnx model.pth --model-name resnet18 --num-classes 4 --input-size 224 224

  # 出力先を指定
  uv run export-onnx model.pth --input-size 224 224 -o output/model.onnx
        """,
    )

    parser.add_argument("model_path", help="変換するPyTorchモデルファイル(.pth)")
    parser.add_argument(
        "--config",
        "-c",
        help="設定ファイルパス（省略時はモデルと同階層のconfig.pyを探索）",
    )
    parser.add_argument(
        "--model-name",
        default="resnet18",
        help="モデル名 (default: resnet18)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        help="分類クラス数（設定ファイルから取得できない場合に必要）",
    )
    parser.add_argument(
        "--input-size",
        nargs=2,
        type=positive_int,
        required=True,
        metavar=("HEIGHT", "WIDTH"),
        help="入力画像サイズ（必須, 1以上）",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="出力ファイルパス (default: 入力ファイルと同じ場所に.onnx拡張子で保存)",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=17,
        help="ONNXオペセットバージョン (default: 17)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="使用デバイス (default: cpu)",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="エクスポート後の検証をスキップ",
    )

    args = parser.parse_args()

    # モデルパスの確認
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"モデルファイルが見つかりません: {model_path}")
        sys.exit(1)

    # 設定ファイルの探索
    config = None
    if args.config:
        config_path = Path(args.config)
    else:
        possible_paths = [
            model_path.parent / "config.py",
            model_path.parent.parent / "config.py",
        ]
        config_path = None
        for p in possible_paths:
            if p.exists():
                config_path = p
                break

    if config_path and config_path.exists():
        try:
            config = ConfigLoader.load_config(str(config_path))
            logger.info(f"設定ファイルを読み込み: {config_path}")
        except Exception as e:
            logger.warning(f"設定ファイルの読み込みに失敗: {e}")
            config = None

    # モデル情報の取得
    model_name = args.model_name
    num_classes = args.num_classes

    if config:
        model_name = config.get("model_name", model_name)
        if num_classes is None:
            num_classes = config.get("num_classes")

    if num_classes is None:
        logger.error("--num-classes を指定するか、設定ファイルを使用してください")
        sys.exit(1)

    # 入力サイズの決定
    input_size = (args.input_size[0], args.input_size[1])

    # 出力パスの決定
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = model_path.with_suffix(".onnx")

    # デバイス設定
    device = torch.device(args.device)

    logger.info(f"モデル: {model_name}")
    logger.info(f"クラス数: {num_classes}")
    logger.info(f"入力サイズ: {input_size[0]}x{input_size[1]}")
    logger.info(f"エクスポートデバイス: {device}")
    logger.info(f"出力先: {output_path}")

    # エクスポーター作成
    try:
        exporter = OnnxExporter(device=device)
        exporter.load_model(model_path, model_name, num_classes)
    except Exception as e:
        logger.error(f"モデルの読み込みに失敗: {e}")
        sys.exit(1)

    # ONNX変換
    try:
        logger.info("ONNX変換を実行中...")
        exporter.export(
            output_path=output_path,
            input_size=input_size,
            opset_version=args.opset_version,
        )
    except Exception as e:
        logger.error(f"ONNX変換に失敗: {e}")
        sys.exit(1)

    # ONNX検証
    if not args.skip_verify:
        logger.info("--- ONNX検証 ---")
        try:
            is_valid = exporter.verify(
                onnx_path=output_path,
                input_size=input_size,
            )
            if is_valid:
                logger.info("検証完了: ONNXモデルは正常です")
            else:
                logger.warning("PyTorchとONNXの出力に差異があります")
                sys.exit(1)
        except Exception as e:
            logger.error(f"ONNX検証に失敗: {e}")
            sys.exit(1)
    else:
        logger.info("検証をスキップしました")


if __name__ == "__main__":
    main()
