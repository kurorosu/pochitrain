#!/usr/bin/env python3
"""TensorRTエンジンを使用した推論CLI.

使用例:
    uv run infer-trt work_dirs/20260118_001/models/model.engine
    uv run infer-trt work_dirs/20260118_001/models/model.engine --data other/val
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
from pochitrain.logging.logger_manager import LogLevel
from pochitrain.pochi_dataset import PochiImageDataset, get_basic_transforms
from pochitrain.utils import (
    get_default_output_base_dir,
    load_config_auto,
    log_inference_result,
    post_process_logits,
    save_classification_report,
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
        description="TensorRTエンジンを使用した高速推論",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本（config・データパスはエンジンパスから自動検出）
  uv run infer-trt work_dirs/20260118_001/models/model.engine

  # データパスを上書き
  uv run infer-trt work_dirs/20260118_001/models/model.engine --data other/val

  # 出力先を上書き
  uv run infer-trt work_dirs/20260118_001/models/model.engine -o results/

前提条件:
  - TensorRT SDKのインストールが必要
  - uv pip install TensorRT-10.x.x.x\\python\\tensorrt-10.x.x-cpXX-win_amd64.whl
        """,
    )

    parser.add_argument("engine_path", help="TensorRTエンジンファイルパス (.engine)")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="デバッグログを有効化",
    )
    parser.add_argument(
        "--data",
        help="推論データディレクトリ（省略時はconfigのval_data_rootを使用）",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="結果出力ディレクトリ（省略時はエンジンパスから自動決定）",
    )

    args = parser.parse_args()

    manager = LoggerManager()
    level = LogLevel.DEBUG if args.debug else LogLevel.INFO
    manager.set_default_level(level)
    manager.set_logger_level(__name__, level)

    # TensorRTの利用可否チェック
    try:
        from pochitrain.tensorrt import TensorRTInference
    except ImportError:
        logger.error(
            "TensorRTがインストールされていません. "
            "TensorRT SDKをインストールしてください."
        )
        sys.exit(1)

    # パス検証
    engine_path = Path(args.engine_path)
    validate_model_path(engine_path)

    # TensorRT推論クラス作成（入力サイズ取得のため先に読み込む）
    inference = TensorRTInference(engine_path)

    # config自動検出・読み込み
    config = load_config_auto(engine_path)

    # データパスの決定（--data指定 or configのval_data_root）
    if args.data:
        data_path = Path(args.data)
    elif "val_data_root" in config:
        data_path = Path(config["val_data_root"])
        logger.debug(f"データパスをconfigから取得: {data_path}")
    else:
        logger.error("--data を指定するか、configにval_data_rootを設定してください")
        sys.exit(1)
    validate_data_path(data_path)

    # 出力ディレクトリの決定
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        base_dir = get_default_output_base_dir(engine_path)
        workspace_manager = InferenceWorkspaceManager(str(base_dir))
        output_dir = workspace_manager.create_workspace()

    # transformの決定（configのval_transformを使用、なければエンジンから自動取得）
    if "val_transform" in config:
        transform = config["val_transform"]
    else:
        # エンジンから入力サイズを自動取得
        engine_input_shape = inference.get_input_shape()
        # shape: (batch, channels, height, width)
        height = engine_input_shape[2]
        width = engine_input_shape[3]
        transform = get_basic_transforms(image_size=height, is_training=False)
        logger.debug(f"入力サイズをエンジンから取得: {height}x{width}")

    logger.debug(f"エンジン: {engine_path}")
    logger.debug(f"データ: {data_path}")
    logger.debug(f"出力先: {output_dir}")

    # データセット作成
    dataset = PochiImageDataset(str(data_path), transform=transform)

    logger.debug(f"クラス: {dataset.get_classes()}")
    logger.debug("使用されたTransform:")
    if hasattr(transform, "transforms"):
        for i, t in enumerate(transform.transforms):
            logger.debug(f"   {i+1}. {t}")

    # ウォームアップ（最初の1枚で10回実行）
    logger.debug("ウォームアップ中...")
    image, _ = dataset[0]
    assert isinstance(image, torch.Tensor)
    image_np = image.numpy()[np.newaxis, ...].astype(np.float32)
    for _ in range(10):
        inference.run(image_np)

    # 推論実行（End-to-End計測の開始）
    logger.info("推論を開始します...")

    # 入力サイズの取得 (TensorRTの入力形状から)
    input_size = None
    try:
        # inference.input_shape から [batch, channel, height, width] を抽出
        shape = inference.input_shape
        if len(shape) == 4:
            input_size = (shape[1], shape[2], shape[3])
    except Exception:
        pass

    e2e_start_time = time.perf_counter()

    all_predictions: List[int] = []
    all_confidences: List[float] = []
    all_true_labels: List[int] = []
    warmup_samples = 1  # 最初の1枚はウォームアップ

    # CUDA Eventで正確な時間計測
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    total_inference_time_ms = 0.0
    total_samples = 0

    for i in range(len(dataset)):
        image, label = dataset[i]
        assert isinstance(image, torch.Tensor)
        image_np = image.numpy()[np.newaxis, ...].astype(np.float32)

        if i == 0:
            # 最初の1枚は計測対象外（ウォームアップ済みだが念のため）
            inference.set_input(image_np)
            inference.execute()
            logits = inference.get_output()
            predicted, confidence = post_process_logits(logits)
        else:
            # 推論時間計測
            inference.set_input(image_np)  # 転送（計測外）

            start_event.record()
            inference.execute()  # 純粋推論のみを計測
            end_event.record()
            torch.cuda.synchronize()
            total_inference_time_ms += start_event.elapsed_time(end_event)
            total_samples += 1

            logits = inference.get_output()  # 取得（計測外）
            predicted, confidence = post_process_logits(logits)

        all_predictions.append(int(predicted[0]))
        all_confidences.append(float(confidence[0]))
        all_true_labels.append(label)

    # 全処理計測の終了
    e2e_total_time_ms = (time.perf_counter() - e2e_start_time) * 1000

    # 精度計算
    correct = sum(p == t for p, t in zip(all_predictions, all_true_labels))
    num_samples = len(dataset)
    avg_time_per_image = (
        total_inference_time_ms / total_samples if total_samples > 0 else 0
    )
    avg_total_time_per_image = e2e_total_time_ms / num_samples if num_samples > 0 else 0

    # 結果ログ出力
    log_inference_result(
        num_samples=num_samples,
        correct=correct,
        avg_time_per_image=avg_time_per_image,
        total_samples=total_samples,
        warmup_samples=warmup_samples,
        avg_total_time_per_image=avg_total_time_per_image,
        input_size=input_size,
    )
    logger.info("推論完了")

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
        filename="tensorrt_inference_results.csv",
    )

    accuracy = (correct / num_samples) * 100 if num_samples > 0 else 0.0

    write_inference_summary(
        output_dir=output_dir,
        model_path=engine_path,
        data_path=data_path,
        num_samples=num_samples,
        accuracy=accuracy,
        avg_time_per_image=avg_time_per_image,
        total_samples=total_samples,
        warmup_samples=warmup_samples,
        avg_total_time_per_image=avg_total_time_per_image,
        input_size=input_size,
        filename="tensorrt_inference_summary.txt",
    )

    logger.info(f"ワークスペース: {output_dir.name}へサマリーファイルを出力しました")

    # 混同行列画像を生成
    cm_config = config.get("confusion_matrix_config", None)
    try:
        save_confusion_matrix_image(
            predicted_labels=all_predictions,
            true_labels=all_true_labels,
            class_names=class_names,
            output_dir=output_dir,
            cm_config=cm_config,
        )
    except Exception as e:
        logger.warning(f"混同行列画像生成に失敗しました: {e}")

    # クラス別精度レポートを生成
    try:
        save_classification_report(
            predicted_labels=all_predictions,
            true_labels=all_true_labels,
            class_names=class_names,
            output_dir=output_dir,
        )
    except Exception as e:
        logger.warning(f"クラス別精度レポート生成に失敗しました: {e}")


if __name__ == "__main__":
    main()
