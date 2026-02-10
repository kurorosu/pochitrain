#!/usr/bin/env python3
"""TensorRTエンジンを使用した推論CLI.

使用例:
    uv run infer-trt work_dirs/20260118_001/models/model.engine
    uv run infer-trt work_dirs/20260118_001/models/model.engine --data other/val
    uv run infer-trt work_dirs/20260118_001/models/model.engine --pipeline gpu
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from pochitrain.logging import LoggerManager
from pochitrain.logging.logger_manager import LogLevel
from pochitrain.pochi_dataset import (
    FastInferenceDataset,
    GpuInferenceDataset,
    PochiImageDataset,
    build_gpu_preprocess_transform,
    convert_transform_for_fast_inference,
    extract_normalize_params,
    get_basic_transforms,
    gpu_normalize,
)
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

PIPELINE_CHOICES = ("auto", "current", "fast", "gpu")


def _resolve_pipeline(requested: str) -> str:
    """--pipeline auto を実際のパイプライン名に解決する.

    TensorRTは常にGPU推論のため, autoはgpuに解決される.

    Args:
        requested: ユーザー指定のパイプライン名

    Returns:
        解決後のパイプライン名 ("current", "fast", "gpu")
    """
    if requested == "auto":
        return "gpu"
    return requested


def _create_dataset_and_params(
    pipeline: str,
    data_path: Path,
    val_transform: Any,
) -> Tuple[PochiImageDataset, str, Optional[List[float]], Optional[List[float]]]:
    """パイプラインに応じたデータセットを作成する.

    Args:
        pipeline: パイプライン名 ("current", "fast", "gpu")
        data_path: データディレクトリパス
        val_transform: configのval_transform

    Returns:
        (dataset, 実際のパイプライン名, mean, std) のタプル.
        mean/stdはgpuパイプライン時のみ値を持つ.
    """
    mean: Optional[List[float]] = None
    std: Optional[List[float]] = None

    if pipeline == "gpu":
        # GPU前処理: Normalize/ToTensor/ConvertDtypeを除外したtransformを構築
        gpu_transform = build_gpu_preprocess_transform(val_transform)
        if gpu_transform is None:
            # PIL専用transformが含まれる場合はcurrentにフォールバック
            logger.info(
                "PIL専用transformが含まれるため, currentパイプラインにフォールバックします"
            )
            dataset: PochiImageDataset = PochiImageDataset(
                str(data_path), transform=val_transform
            )
            return dataset, "current", None, None

        try:
            mean, std = extract_normalize_params(val_transform)
        except ValueError:
            logger.warning(
                "Normalizeが見つからないため, fastパイプラインにフォールバックします"
            )
            fast_transform = convert_transform_for_fast_inference(val_transform)
            if fast_transform is not None:
                dataset = FastInferenceDataset(str(data_path), transform=fast_transform)
                return dataset, "fast", None, None
            dataset = PochiImageDataset(str(data_path), transform=val_transform)
            return dataset, "current", None, None

        dataset = GpuInferenceDataset(
            str(data_path),
            transform=gpu_transform if gpu_transform.transforms else None,
        )
        return dataset, "gpu", mean, std

    if pipeline == "fast":
        fast_transform = convert_transform_for_fast_inference(val_transform)
        if fast_transform is not None:
            dataset = FastInferenceDataset(str(data_path), transform=fast_transform)
            return dataset, "fast", None, None
        logger.info("PochiImageDatasetにフォールバックします")
        dataset = PochiImageDataset(str(data_path), transform=val_transform)
        return dataset, "current", None, None

    # current
    dataset = PochiImageDataset(str(data_path), transform=val_transform)
    return dataset, "current", None, None


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

  # パイプライン指定
  uv run infer-trt model.engine --pipeline gpu     # GPU前処理 (デフォルト)
  uv run infer-trt model.engine --pipeline fast     # CPU最適化 (Plan A)
  uv run infer-trt model.engine --pipeline current  # 従来 (PIL)

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
    parser.add_argument(
        "--pipeline",
        choices=PIPELINE_CHOICES,
        default="auto",
        help="前処理パイプライン: auto(デフォルト), current(PIL), fast(CPU最適化), gpu(GPU前処理)",
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
        val_transform = config["val_transform"]
    else:
        # エンジンから入力サイズを自動取得
        engine_input_shape = inference.get_input_shape()
        # shape: (batch, channels, height, width)
        height = engine_input_shape[2]
        width = engine_input_shape[3]
        val_transform = get_basic_transforms(image_size=height, is_training=False)
        logger.debug(f"入力サイズをエンジンから取得: {height}x{width}")

    # パイプライン解決
    pipeline = _resolve_pipeline(args.pipeline)

    # configからパラメータ取得
    num_workers = config.get("num_workers", 0)
    pin_memory = config.get("pin_memory", True)

    # データセット作成
    dataset, pipeline, norm_mean, norm_std = _create_dataset_and_params(
        pipeline, data_path, val_transform
    )
    use_gpu_pipeline = pipeline == "gpu"

    logger.debug(f"エンジン: {engine_path}")
    logger.debug(f"データ: {data_path}")
    logger.debug(f"ワーカー数: {num_workers}")
    logger.debug(f"パイプライン: {pipeline}")
    logger.debug(f"出力先: {output_dir}")

    # DataLoader作成（TensorRTはbatch_size=1のみ対応）
    data_loader: DataLoader[Any] = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # 使用されたtransformをログ出力
    if dataset.transform is not None:
        logger.debug(f"クラス: {dataset.get_classes()}")
        logger.debug("使用されたTransform:")
        if hasattr(dataset.transform, "transforms"):
            for i, t in enumerate(dataset.transform.transforms):
                logger.debug(f"   {i+1}. {t}")

    # ウォームアップ（最初の1枚で10回実行）
    logger.debug("ウォームアップ中...")
    image, _ = dataset[0]
    assert isinstance(image, torch.Tensor)

    if use_gpu_pipeline:
        assert norm_mean is not None and norm_std is not None
        warmup_gpu = gpu_normalize(image, norm_mean, norm_std)
        for _ in range(10):
            inference.set_input_gpu(warmup_gpu)
            inference.execute()
            inference.get_output()
    else:
        image_np = image.numpy()[np.newaxis, ...]
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

    for i, (image, label) in enumerate(data_loader):
        # 前処理: パイプラインに応じた入力準備
        if use_gpu_pipeline:
            assert norm_mean is not None and norm_std is not None
            gpu_tensor = gpu_normalize(image, norm_mean, norm_std)

        if i == 0:
            # 最初の1枚は計測対象外（ウォームアップ済みだが念のため）
            if use_gpu_pipeline:
                inference.set_input_gpu(gpu_tensor)
            else:
                inference.set_input(image.numpy())
            inference.execute()
            logits = inference.get_output()
            predicted, confidence = post_process_logits(logits)
        else:
            # 転送（計測外）
            if use_gpu_pipeline:
                inference.set_input_gpu(gpu_tensor)
            else:
                inference.set_input(image.numpy())

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
        all_true_labels.append(label.item())

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
        extra_info={"パイプライン": pipeline},
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
