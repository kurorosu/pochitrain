#!/usr/bin/env python3
"""ONNXモデルを使用した推論CLI.

使用例:
    uv run infer-onnx model.onnx --data data/val --input-size 224 224
    uv run infer-onnx model.onnx --data data/val --config config.py
    uv run infer-onnx model.onnx --data data/val --input-size 224 224 --gpu
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

from pochitrain.logging import LoggerManager
from pochitrain.onnx import OnnxInference
from pochitrain.pochi_dataset import PochiImageDataset, get_basic_transforms
from pochitrain.utils import ConfigLoader

logger: logging.Logger = LoggerManager().get_logger(__name__)


def main() -> None:
    """メイン関数."""
    parser = argparse.ArgumentParser(
        description="ONNXモデルを使用した推論",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本的な推論
  uv run infer-onnx model.onnx --data data/val --input-size 224 224

  # 設定ファイルを使用
  uv run infer-onnx model.onnx --data data/val --config config.py

  # GPU使用
  uv run infer-onnx model.onnx --data data/val --input-size 224 224 --gpu

  # 結果を保存
  uv run infer-onnx model.onnx --data data/val --input-size 224 224 -o results/
        """,
    )

    parser.add_argument("model_path", help="ONNXモデルファイルパス")
    parser.add_argument(
        "--data",
        required=True,
        help="推論データディレクトリ",
    )
    parser.add_argument(
        "--config",
        "-c",
        help="設定ファイルパス（変換設定を取得）",
    )
    parser.add_argument(
        "--input-size",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        help="入力画像サイズ（configがない場合は必須）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="バッチサイズ (default: 1)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="GPUを使用（onnxruntime-gpuが必要）",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="結果出力ディレクトリ",
    )

    args = parser.parse_args()

    # モデルパスの確認
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"モデルファイルが見つかりません: {model_path}")
        sys.exit(1)

    # データパスの確認
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"データディレクトリが見つかりません: {data_path}")
        sys.exit(1)

    # 設定ファイルの読み込み
    config = None
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            try:
                config = ConfigLoader.load_config(str(config_path))
                logger.info(f"設定ファイルを読み込み: {config_path}")
            except Exception as e:
                logger.warning(f"設定ファイルの読み込みに失敗: {e}")

    # 入力サイズの決定
    if args.input_size:
        input_size = (args.input_size[0], args.input_size[1])
    elif config and "val_transform" in config:
        input_size = (224, 224)
        logger.warning("入力サイズを224x224と仮定しています")
    else:
        logger.error("--input-size を指定するか、--config を使用してください")
        sys.exit(1)

    batch_size = args.batch_size

    logger.info(f"モデル: {model_path}")
    logger.info(f"データ: {data_path}")
    logger.info(f"入力サイズ: {input_size[0]}x{input_size[1]}")
    logger.info(f"バッチサイズ: {batch_size}")
    logger.info(f"GPU使用: {args.gpu}")

    # データセット作成
    transform = get_basic_transforms(image_size=input_size[0], is_training=False)
    dataset = PochiImageDataset(str(data_path), transform=transform)

    logger.info(f"データセット: {len(dataset)}枚")
    logger.info(f"クラス: {dataset.get_classes()}")

    # ONNX推論クラス作成
    logger.info("ONNXセッションを作成中...")
    inference = OnnxInference(model_path, use_gpu=args.gpu)

    # 推論実行
    logger.info("推論を開始...")
    all_predictions: List[int] = []
    all_confidences: List[float] = []
    all_true_labels: List[int] = []
    total_inference_time = 0.0
    total_samples = 0

    num_batches = (len(dataset) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(dataset))

        batch_images = []
        batch_labels = []
        for i in range(start_idx, end_idx):
            image, label = dataset[i]
            batch_images.append(image.numpy())
            batch_labels.append(label)

        images_np = np.stack(batch_images).astype(np.float32)

        start_time = time.perf_counter()
        predicted, confidence = inference.run(images_np)
        inference_time = (time.perf_counter() - start_time) * 1000

        total_inference_time += inference_time
        total_samples += len(batch_images)

        all_predictions.extend(predicted.tolist())
        all_confidences.extend(confidence.tolist())
        all_true_labels.extend(batch_labels)

    # 精度計算
    correct = sum(p == t for p, t in zip(all_predictions, all_true_labels))
    accuracy = (correct / total_samples) * 100 if total_samples > 0 else 0.0
    avg_time_per_image = (
        total_inference_time / total_samples if total_samples > 0 else 0
    )

    logger.info("推論完了")
    logger.info(f"精度: {correct}/{total_samples} ({accuracy:.2f}%)")
    logger.info(f"平均推論時間: {avg_time_per_image:.2f} ms/image")
    logger.info(f"総推論時間: {total_inference_time:.2f} ms")

    # 結果出力
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # CSV出力
        csv_path = output_dir / "onnx_inference_results.csv"
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
        summary_path = output_dir / "onnx_inference_summary.txt"
        providers = inference.get_providers()
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"モデル: {model_path}\n")
            f.write(f"データ: {data_path}\n")
            f.write(f"サンプル数: {total_samples}\n")
            f.write(f"精度: {accuracy:.2f}%\n")
            f.write(f"平均推論時間: {avg_time_per_image:.2f} ms/image\n")
            f.write(f"実行プロバイダー: {providers}\n")

        logger.info(f"サマリーを保存: {summary_path}")


if __name__ == "__main__":
    main()
