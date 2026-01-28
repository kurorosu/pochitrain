#!/usr/bin/env python3
"""TensorRTエンジンを使用した推論CLI.

使用例:
    uv run infer-trt model.engine --data data/val
    uv run infer-trt model.engine --data data/val --config config.py
    uv run infer-trt model.engine --data data/val -o results/
"""

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from pochitrain.logging import LoggerManager
from pochitrain.pochi_dataset import PochiImageDataset, get_basic_transforms
from pochitrain.utils import ConfigLoader

logger: logging.Logger = LoggerManager().get_logger(__name__)


def main() -> None:
    """メイン関数."""
    parser = argparse.ArgumentParser(
        description="TensorRTエンジンを使用した高速推論",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本（入力サイズはエンジンから自動取得）
  uv run infer-trt model.engine --data data/val

  # 設定ファイルを使用（カスタムtransformが必要な場合）
  uv run infer-trt model.engine --data data/val --config config.py

  # 結果を保存
  uv run infer-trt model.engine --data data/val -o results/

前提条件:
  - TensorRT SDKのインストールが必要
  - uv pip install TensorRT-10.x.x.x\\python\\tensorrt-10.x.x-cpXX-win_amd64.whl
        """,
    )

    parser.add_argument("engine_path", help="TensorRTエンジンファイルパス (.engine)")
    parser.add_argument(
        "--data",
        required=True,
        help="推論データディレクトリ",
    )
    parser.add_argument(
        "--config",
        "-c",
        help="設定ファイルパス（transformを取得）",
    )
    parser.add_argument(
        "--input-size",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        help="入力画像サイズ（省略時はエンジンから自動取得）",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="結果出力ディレクトリ",
    )

    args = parser.parse_args()

    # TensorRTの利用可否チェック
    try:
        from pochitrain.tensorrt import TensorRTInference
    except ImportError:
        logger.error(
            "TensorRTがインストールされていません. "
            "TensorRT SDKをインストールしてください."
        )
        sys.exit(1)

    # エンジンパスの確認
    engine_path = Path(args.engine_path)
    if not engine_path.exists():
        logger.error(f"エンジンファイルが見つかりません: {engine_path}")
        sys.exit(1)

    # データパスの確認
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"データディレクトリが見つかりません: {data_path}")
        sys.exit(1)

    # TensorRT推論クラス作成（入力サイズ取得のため先に読み込む）
    logger.info("TensorRTエンジンを読み込み中...")
    inference = TensorRTInference(engine_path)

    # 設定ファイルの読み込み
    config: Optional[Dict[str, Any]] = None
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            try:
                config = ConfigLoader.load_config(str(config_path))
                logger.info(f"設定ファイルを読み込み: {config_path}")
            except Exception as e:
                logger.warning(f"設定ファイルの読み込みに失敗: {e}")
        else:
            logger.warning(f"設定ファイルが見つかりません: {config_path}")

    # transformの決定
    transform = None
    input_size_str = ""

    if config and "val_transform" in config:
        # configからval_transformを使用
        transform = config["val_transform"]
        logger.info("val_transformを設定ファイルから取得")
        input_size_str = "config指定"
    elif args.input_size:
        # --input-sizeからtransformを生成
        input_size = (args.input_size[0], args.input_size[1])
        transform = get_basic_transforms(image_size=input_size[0], is_training=False)
        input_size_str = f"{input_size[0]}x{input_size[1]}"
    else:
        # エンジンから入力サイズを自動取得
        engine_input_shape = inference.get_input_shape()
        # shape: (batch, channels, height, width)
        height = engine_input_shape[2]
        width = engine_input_shape[3]
        transform = get_basic_transforms(image_size=height, is_training=False)
        input_size_str = f"{height}x{width} (エンジンから自動取得)"
        logger.info(f"入力サイズをエンジンから取得: {height}x{width}")

    logger.info(f"エンジン: {engine_path}")
    logger.info(f"データ: {data_path}")
    logger.info(f"入力サイズ: {input_size_str}")

    # データセット作成
    dataset = PochiImageDataset(str(data_path), transform=transform)

    logger.info(f"データセット: {len(dataset)}枚")
    logger.info(f"クラス: {dataset.get_classes()}")

    # ウォームアップ（最初の1枚で10回実行）
    logger.info("ウォームアップ中...")
    image, _ = dataset[0]
    image_np = image.numpy()[np.newaxis, ...].astype(np.float32)
    for _ in range(10):
        inference.run(image_np)

    # 推論実行
    logger.info("推論を開始...")
    all_predictions: List[int] = []
    all_confidences: List[float] = []
    all_true_labels: List[int] = []
    warmup_samples = 1  # 最初の1枚はウォームアップ

    # CUDA Eventで正確な時間計測
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    inference_time_ms = 0.0
    total_samples = 0

    for i in range(len(dataset)):
        image, label = dataset[i]
        image_np = image.numpy()[np.newaxis, ...].astype(np.float32)

        if i == 0:
            # 最初の1枚は計測対象外（ウォームアップ済みだが念のため）
            predicted, confidence = inference.run(image_np)
        else:
            # 推論時間計測
            start_event.record()
            predicted, confidence = inference.run(image_np)
            end_event.record()
            torch.cuda.synchronize()
            inference_time_ms += start_event.elapsed_time(end_event)
            total_samples += 1

        all_predictions.append(int(predicted[0]))
        all_confidences.append(float(confidence[0]))
        all_true_labels.append(label)

    # 精度計算
    correct = sum(p == t for p, t in zip(all_predictions, all_true_labels))
    accuracy = (correct / len(dataset)) * 100 if len(dataset) > 0 else 0.0
    avg_time_per_image = inference_time_ms / total_samples if total_samples > 0 else 0

    logger.info("推論完了")
    logger.info(f"精度: {correct}/{len(dataset)} ({accuracy:.2f}%)")
    logger.info(
        f"平均推論時間: {avg_time_per_image:.2f} ms/image "
        f"(計測: {total_samples}枚, ウォームアップ除外: {warmup_samples}枚)"
    )
    logger.info(f"スループット: {1000 / avg_time_per_image:.1f} images/sec")

    # 結果出力
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # CSV出力
        csv_path = output_dir / "tensorrt_inference_results.csv"
        class_names = dataset.get_classes()
        image_paths = dataset.get_file_paths()

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "image_path",
                    "predicted",
                    "predicted_class",
                    "true",
                    "true_class",
                    "confidence",
                    "correct",
                ]
            )
            for path, pred, true, conf in zip(
                image_paths, all_predictions, all_true_labels, all_confidences
            ):
                writer.writerow(
                    [
                        path,
                        pred,
                        class_names[pred],
                        true,
                        class_names[true],
                        f"{conf:.4f}",
                        pred == true,
                    ]
                )

        logger.info(f"結果を保存: {csv_path}")

        # サマリー出力
        summary_path = output_dir / "tensorrt_inference_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"エンジン: {engine_path}\n")
            f.write(f"データ: {data_path}\n")
            f.write(f"入力サイズ: {input_size_str}\n")
            f.write(f"サンプル数: {len(dataset)}\n")
            f.write(f"精度: {accuracy:.2f}%\n")
            f.write(f"平均推論時間: {avg_time_per_image:.2f} ms/image\n")
            f.write(f"スループット: {1000 / avg_time_per_image:.1f} images/sec\n")

        logger.info(f"サマリーを保存: {summary_path}")


if __name__ == "__main__":
    main()
