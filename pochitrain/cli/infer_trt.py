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
from pathlib import Path
from typing import Any, List, Optional, Tuple

from torch.utils.data import DataLoader

from pochitrain.inference.adapters.trt_runtime_adapter import TensorRTRuntimeAdapter
from pochitrain.inference.pipeline_strategy import (
    create_dataset_and_params as shared_create_dataset_and_params,
)
from pochitrain.inference.services.execution_service import ExecutionService
from pochitrain.inference.services.result_export_service import ResultExportService
from pochitrain.inference.services.trt_inference_service import TensorRTInferenceService
from pochitrain.inference.types.execution_types import ExecutionRequest
from pochitrain.inference.types.orchestration_types import InferenceCliRequest
from pochitrain.inference.types.result_export_types import ResultExportRequest
from pochitrain.logging import LoggerManager
from pochitrain.logging.logger_manager import LogLevel
from pochitrain.pochi_dataset import (
    PochiImageDataset,
    create_scaled_normalize_tensors,
    get_basic_transforms,
)
from pochitrain.utils import (
    load_config_auto,
    log_inference_result,
    validate_model_path,
)

logger: logging.Logger = LoggerManager().get_logger(__name__)

PIPELINE_CHOICES = ("auto", "current", "fast", "gpu")


def _create_dataset_and_params(
    pipeline: str,
    data_path: Path,
    val_transform: Any,
) -> Tuple[PochiImageDataset, str, Optional[List[float]], Optional[List[float]]]:
    """パイプライン別データセット生成を共通処理へ委譲する.

    Args:
        pipeline: パイプライン名.
        data_path: 推論対象データディレクトリ.
        val_transform: 検証用 transform.

    Returns:
        データセット, 実際に適用されたパイプライン名, mean, std.
    """
    return shared_create_dataset_and_params(pipeline, data_path, val_transform)


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
    orchestration_service = TensorRTInferenceService()

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

    cli_request = InferenceCliRequest(
        model_path=engine_path,
        data_path=Path(args.data) if args.data else None,
        output_dir=Path(args.output) if args.output else None,
        requested_pipeline=args.pipeline,
    )
    try:
        resolved_paths = orchestration_service.resolve_paths(cli_request, config)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    data_path = resolved_paths.data_path
    output_dir = resolved_paths.output_dir

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
    pipeline = orchestration_service.resolve_pipeline(args.pipeline)
    runtime_options = orchestration_service.resolve_runtime_options(config, pipeline)
    num_workers = runtime_options.num_workers
    pin_memory = runtime_options.pin_memory

    # データセット作成
    dataset, pipeline, norm_mean, norm_std = _create_dataset_and_params(
        pipeline, data_path, val_transform
    )
    use_gpu_pipeline = runtime_options.use_gpu_pipeline

    logger.debug(f"エンジン: {engine_path}")
    logger.debug(f"データ: {data_path}")
    logger.debug(f"ワーカー数: {num_workers}")
    logger.debug(f"パイプライン: {pipeline}")
    logger.debug(f"出力先: {output_dir}")

    # DataLoader作成（TensorRTはbatch_size=1のみ対応）
    data_loader: DataLoader[Any] = DataLoader(
        dataset,
        batch_size=runtime_options.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    mean_255 = None
    std_255 = None
    if use_gpu_pipeline:
        assert norm_mean is not None and norm_std is not None
        mean_255, std_255 = create_scaled_normalize_tensors(norm_mean, norm_std)

    # 使用されたtransformをログ出力
    if dataset.transform is not None:
        logger.debug(f"クラス: {dataset.get_classes()}")
        logger.debug("使用されたTransform:")
        if hasattr(dataset.transform, "transforms"):
            for i, t in enumerate(dataset.transform.transforms):
                logger.debug(f"   {i+1}. {t}")

    # 推論実行（End-to-End計測の開始）
    logger.info("推論を開始します...")

    # 入力サイズの取得 (TensorRTの入力形状から)
    input_size = None
    try:
        # inference.input_shape から [batch, channel, height, width] を抽出
        shape = inference.input_shape
        input_size = orchestration_service.resolve_input_size(shape)
    except Exception:
        pass

    execution_request = ExecutionRequest(
        use_gpu_pipeline=use_gpu_pipeline,
        mean_255=mean_255,
        std_255=std_255,
        warmup_repeats=10,
        skip_measurement_batches=1,
        use_cuda_timing=True,
        gpu_non_blocking=bool(config.get("gpu_non_blocking", True)),
    )
    logger.debug("ウォームアップ中...")
    execution_service = ExecutionService()
    runtime_adapter = TensorRTRuntimeAdapter(inference)
    execution_result = execution_service.run(
        data_loader=data_loader,
        runtime=runtime_adapter,
        request=execution_request,
    )
    all_predictions = execution_result.predictions
    all_confidences = execution_result.confidences
    all_true_labels = execution_result.true_labels
    total_inference_time_ms = execution_result.total_inference_time_ms
    total_samples = execution_result.total_samples
    warmup_samples = execution_result.warmup_samples
    e2e_total_time_ms = execution_result.e2e_total_time_ms

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
    export_service = ResultExportService(logger)
    export_service.export(
        ResultExportRequest(
            output_dir=output_dir,
            model_path=engine_path,
            data_path=data_path,
            image_paths=image_paths,
            predictions=all_predictions,
            true_labels=all_true_labels,
            confidences=all_confidences,
            class_names=class_names,
            num_samples=num_samples,
            correct=correct,
            avg_time_per_image=avg_time_per_image,
            total_samples=total_samples,
            warmup_samples=warmup_samples,
            avg_total_time_per_image=avg_total_time_per_image,
            input_size=input_size,
            results_filename="tensorrt_inference_results.csv",
            summary_filename="tensorrt_inference_summary.txt",
            extra_info={"パイプライン": pipeline},
            cm_config=config.get("confusion_matrix_config", None),
        )
    )

    logger.info(f"ワークスペース: {output_dir.name}にサマリーファイルを出力しました")


if __name__ == "__main__":
    main()
