#!/usr/bin/env python3
"""ONNXモデルを使用した推論CLI.

使用例:
    uv run infer-onnx work_dirs/20260118_001/models/model.onnx
    uv run infer-onnx work_dirs/20260118_001/models/model.onnx --data other/val
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

from pochitrain.logging import LoggerManager
from pochitrain.onnx import OnnxInference
from pochitrain.pochi_dataset import PochiImageDataset
from pochitrain.utils import (
    get_default_output_base_dir,
    load_config_auto,
    log_inference_result,
    save_confusion_matrix_image,
    validate_data_path,
    validate_model_path,
    write_inference_csv,
    write_inference_summary,
)
from pochitrain.utils.directory_manager import InferenceWorkspaceManager

logger: logging.Logger = LoggerManager().get_logger(__name__)


def main() -> None:
    """メイン関数."""
    parser = argparse.ArgumentParser(
        description="ONNXモデルを使用した推論",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本（config・データパスはモデルパスから自動検出）
  uv run infer-onnx work_dirs/20260118_001/models/model.onnx

  # データパスを上書き
  uv run infer-onnx work_dirs/20260118_001/models/model.onnx --data other/val

  # 出力先を上書き
  uv run infer-onnx work_dirs/20260118_001/models/model.onnx -o results/
        """,
    )

    parser.add_argument("model_path", help="ONNXモデルファイルパス")
    parser.add_argument(
        "--data",
        help="推論データディレクトリ（省略時はconfigのval_data_rootを使用）",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="結果出力ディレクトリ（省略時はモデルパスから自動決定）",
    )

    args = parser.parse_args()

    # パス検証
    model_path = Path(args.model_path)
    validate_model_path(model_path)

    # config自動検出・読み込み
    config = load_config_auto(model_path)

    # データパスの決定（--data指定 or configのval_data_root）
    if args.data:
        data_path = Path(args.data)
    elif "val_data_root" in config:
        data_path = Path(config["val_data_root"])
        logger.info(f"データパスをconfigから取得: {data_path}")
    else:
        logger.error("--data を指定するか、configにval_data_rootを設定してください")
        sys.exit(1)
    validate_data_path(data_path)

    # 出力ディレクトリの決定
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        base_dir = get_default_output_base_dir(model_path)
        manager = InferenceWorkspaceManager(str(base_dir))
        output_dir = manager.create_workspace()

    # configからパラメータ取得
    batch_size = config.get("batch_size", 1)
    use_gpu = config.get("device", "cpu") == "cuda"
    val_transform = config["val_transform"]

    logger.info(f"モデル: {model_path}")
    logger.info(f"データ: {data_path}")
    logger.info(f"バッチサイズ: {batch_size}")
    logger.info(f"GPU使用: {use_gpu}")
    logger.info(f"出力先: {output_dir}")

    # データセット作成（configのval_transformを使用）
    dataset = PochiImageDataset(str(data_path), transform=val_transform)

    logger.info(f"データセット: {len(dataset)}枚")
    logger.info(f"クラス: {dataset.get_classes()}")

    # ONNX推論クラス作成
    logger.info("ONNXセッションを作成中...")
    inference = OnnxInference(model_path, use_gpu=use_gpu)

    # ウォームアップ（最初の1バッチで10回実行）
    logger.info("ウォームアップ中...")
    warmup_image, _ = dataset[0]
    warmup_np = warmup_image.numpy()[np.newaxis, ...].astype(np.float32)
    for _ in range(10):
        inference.run(warmup_np)

    # 推論実行
    logger.info("推論を開始...")
    all_predictions: List[int] = []
    all_confidences: List[float] = []
    all_true_labels: List[int] = []
    total_inference_time_ms = 0.0
    total_samples = 0
    warmup_samples = 0

    num_batches = (len(dataset) + batch_size - 1) // batch_size

    # GPU時間計測用のCUDA Event
    if use_gpu:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

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

        if batch_idx == 0:
            # 最初のバッチは計測対象外（ウォームアップ）
            predicted, confidence = inference.run(images_np)
            warmup_samples = len(batch_images)
        else:
            # 推論時間計測
            if use_gpu:
                start_event.record()
                predicted, confidence = inference.run(images_np)
                end_event.record()
                torch.cuda.synchronize()
                inference_time_ms = start_event.elapsed_time(end_event)
            else:
                start_time = time.perf_counter()
                predicted, confidence = inference.run(images_np)
                inference_time_ms = (time.perf_counter() - start_time) * 1000

            total_inference_time_ms += inference_time_ms
            total_samples += len(batch_images)

        all_predictions.extend(predicted.tolist())
        all_confidences.extend(confidence.tolist())
        all_true_labels.extend(batch_labels)

    # 精度計算
    correct = sum(p == t for p, t in zip(all_predictions, all_true_labels))
    num_samples = len(dataset)
    avg_time_per_image = (
        total_inference_time_ms / total_samples if total_samples > 0 else 0
    )

    # 結果ログ出力
    log_inference_result(
        num_samples=num_samples,
        correct=correct,
        avg_time_per_image=avg_time_per_image,
        total_samples=total_samples,
        warmup_samples=warmup_samples,
    )

    # 結果ファイル出力
    class_names = dataset.get_classes()
    image_paths = dataset.get_file_paths()

    write_inference_csv(
        output_dir=output_dir,
        image_paths=image_paths,
        predictions=all_predictions,
        true_labels=all_true_labels,
        confidences=all_confidences,
        class_names=class_names,
        filename="onnx_inference_results.csv",
    )

    accuracy = (correct / num_samples) * 100 if num_samples > 0 else 0.0
    providers = inference.get_providers()

    write_inference_summary(
        output_dir=output_dir,
        model_path=model_path,
        data_path=data_path,
        num_samples=num_samples,
        accuracy=accuracy,
        avg_time_per_image=avg_time_per_image,
        total_samples=total_samples,
        warmup_samples=warmup_samples,
        filename="onnx_inference_summary.txt",
        extra_info={"実行プロバイダー": providers},
    )

    # 混同行列画像を生成
    cm_config = config.get("confusion_matrix_config", None)
    try:
        cm_path = save_confusion_matrix_image(
            predicted_labels=all_predictions,
            true_labels=all_true_labels,
            class_names=class_names,
            output_dir=output_dir,
            cm_config=cm_config,
        )
        logger.info(f"混同行列画像を生成しました: {cm_path}")
    except Exception as e:
        logger.warning(f"混同行列画像生成に失敗しました: {e}")


if __name__ == "__main__":
    main()
