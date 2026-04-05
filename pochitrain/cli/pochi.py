#!/usr/bin/env python3
"""pochitrain 統一CLI エントリーポイント.

訓練・推論・ハイパーパラメータ最適化・TensorRT変換を統合したコマンドライン インターフェース.
"""

import argparse
import sys

from pochitrain.cli.arg_types import positive_int
from pochitrain.cli.commands.convert import convert_command
from pochitrain.cli.commands.infer import infer_command
from pochitrain.cli.commands.optimize import optimize_command
from pochitrain.cli.commands.serve import serve_command
from pochitrain.cli.commands.train import train_command


def main() -> None:
    """メイン関数."""
    parser = argparse.ArgumentParser(
        description="pochitrain - 統合CLI（訓練・推論・最適化・変換）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  訓練
  uv run pochi train --config configs/pochi_train_config.py

  推論（基本）
  uv run pochi infer work_dirs/20250813_003/models/best_epoch40.pth

  推論（データ・設定を指定）
  uv run pochi infer work_dirs/20250813_003/models/best_epoch40.pth
    -d data/val -c work_dirs/20250813_003/config.py

  推論（カスタム出力先）
  uv run pochi infer work_dirs/20250813_003/models/best_epoch40.pth
    --data data/test --config-path work_dirs/20250813_003/config.py
    --output custom_results

  ハイパーパラメータ最適化
  uv run pochi optimize --config configs/pochi_train_config.py

  TensorRT変換（INT8量子化）
  uv run pochi convert model.onnx --int8

  TensorRT変換（FP16）
  uv run pochi convert model.onnx --fp16

  TensorRT変換（キャリブレーションデータ指定）
  uv run pochi convert model.onnx --int8 --calib-data data/val

  推論APIサーバー起動
  uv run pochi serve work_dirs/20260206_001/models/best_epoch40.pth

  推論APIサーバー起動（設定指定）
  uv run pochi serve work_dirs/20260206_001/models/best_epoch40.pth
    -c work_dirs/20260206_001/config.py --host 0.0.0.0 --port 8000
        """,
    )

    parser.add_argument("--debug", action="store_true", help="DEBUGログを有効化")

    subparsers = parser.add_subparsers(dest="command", help="サブコマンド")

    train_parser = subparsers.add_parser("train", help="モデル訓練")
    train_parser.add_argument(
        "--config",
        default="configs/pochi_train_config.py",
        help="設定ファイルパス (default: configs/pochi_train_config.py)",
    )

    infer_parser = subparsers.add_parser("infer", help="モデル推論")
    infer_parser.add_argument("model_path", help="モデルファイルパス")
    infer_parser.add_argument(
        "--data", "-d", help="推論データパス（省略時はconfigのval_data_rootを使用）"
    )
    infer_parser.add_argument(
        "--config-path",
        "-c",
        help="設定ファイルパス（省略時はモデルパスから自動検出）",
    )
    infer_parser.add_argument(
        "--output",
        "-o",
        help="結果出力ディレクトリ（default: モデルと同じディレクトリ/inference_results）",
    )
    infer_parser.add_argument(
        "--pipeline",
        choices=("auto", "current", "fast", "gpu"),
        default="current",
        help="前処理パイプライン: current(デフォルト/PIL), fast(CPU最適化), gpu(GPU前処理). PyTorch推論では current のみ対応.",
    )
    infer_parser.add_argument(
        "--benchmark-json",
        action="store_true",
        help="ベンチマーク結果を benchmark_result.json として出力する",
    )
    infer_parser.add_argument(
        "--benchmark-env-name",
        default=None,
        help="ベンチマーク結果の環境ラベル（省略時は自動決定）",
    )

    optimize_parser = subparsers.add_parser("optimize", help="ハイパーパラメータ最適化")
    optimize_parser.add_argument(
        "--config",
        default="configs/pochi_train_config.py",
        help="設定ファイルパス (default: configs/pochi_train_config.py)",
    )
    optimize_parser.add_argument(
        "--output",
        "-o",
        default="work_dirs/optuna_results",
        help="結果出力ディレクトリ (default: work_dirs/optuna_results)",
    )

    serve_parser = subparsers.add_parser("serve", help="推論APIサーバー起動")
    serve_parser.add_argument("model_path", help="モデルファイルパス")
    serve_parser.add_argument(
        "--config-path",
        "-c",
        help="設定ファイルパス（省略時はモデルパスから自動検出）",
    )
    serve_parser.add_argument(
        "--backend",
        choices=("pytorch",),
        default="pytorch",
        help="推論バックエンド (default: pytorch)",
    )
    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="バインドホスト (default: 127.0.0.1)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="バインドポート (default: 8000)",
    )

    convert_parser = subparsers.add_parser(
        "convert", help="ONNXモデルをTensorRTエンジンに変換"
    )
    convert_parser.add_argument("onnx_path", help="ONNXモデルファイルパス (.onnx)")
    convert_parser.add_argument(
        "--fp16",
        action="store_true",
        help="FP16精度で変換",
    )
    convert_parser.add_argument(
        "--int8",
        action="store_true",
        help="INT8精度で変換 (キャリブレーションが必要)",
    )
    convert_parser.add_argument(
        "--output",
        "-o",
        help="出力エンジンファイルパス (default: 入力ファイルと同じ場所に.engine拡張子)",
    )
    convert_parser.add_argument(
        "--config-path",
        "-c",
        help="設定ファイルパス (INT8時にval_transformとデータパスを取得, "
        "省略時はONNXパスから自動検出)",
    )
    convert_parser.add_argument(
        "--calib-data",
        help="キャリブレーションデータディレクトリ "
        "(INT8時に使用, 省略時はconfigのval_data_rootを使用)",
    )
    convert_parser.add_argument(
        "--input-size",
        nargs=2,
        type=positive_int,
        metavar=("HEIGHT", "WIDTH"),
        help="入力画像サイズ (動的シェイプONNXモデルの変換時に必要, 1以上)",
    )
    convert_parser.add_argument(
        "--calib-samples",
        type=positive_int,
        default=500,
        help="キャリブレーションサンプル数 (default: 500, 1以上)",
    )
    convert_parser.add_argument(
        "--calib-batch-size",
        type=positive_int,
        default=1,
        help="キャリブレーションバッチサイズ (default: 1, 1以上)",
    )
    convert_parser.add_argument(
        "--workspace-size",
        type=positive_int,
        default=1 << 30,
        help="TensorRTワークスペースサイズ (bytes, default: 1GB, 1以上)",
    )

    args = parser.parse_args()

    if args.command == "train":
        train_command(args)
    elif args.command == "infer":
        infer_command(args)
    elif args.command == "optimize":
        optimize_command(args)
    elif args.command == "convert":
        convert_command(args)
    elif args.command == "serve":
        serve_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
