#!/usr/bin/env python3
"""ONNXモデルを使用した推論CLI.

使用例:
    uv run infer-onnx work_dirs/20260118_001/models/model.onnx
    uv run infer-onnx work_dirs/20260118_001/models/model.onnx --data other/val
    uv run infer-onnx work_dirs/20260118_001/models/model.onnx --pipeline gpu
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

from torch.utils.data import DataLoader

from pochitrain.inference.adapters.onnx_runtime_adapter import OnnxRuntimeAdapter
from pochitrain.inference.benchmark import (
    build_onnx_benchmark_result,
    resolve_env_name,
    write_benchmark_result_json,
)
from pochitrain.inference.pipeline_strategy import (
    create_dataset_and_params as shared_create_dataset_and_params,
)
from pochitrain.inference.services.execution_service import ExecutionService
from pochitrain.inference.services.onnx_inference_service import OnnxInferenceService
from pochitrain.inference.services.result_export_service import ResultExportService
from pochitrain.inference.types.execution_types import ExecutionRequest
from pochitrain.inference.types.orchestration_types import InferenceCliRequest
from pochitrain.inference.types.result_export_types import ResultExportRequest
from pochitrain.logging import LoggerManager
from pochitrain.logging.logger_manager import LogLevel
from pochitrain.onnx import OnnxInference
from pochitrain.pochi_dataset import (
    PochiImageDataset,
    create_scaled_normalize_tensors,
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

  # パイプライン指定
  uv run infer-onnx model.onnx --pipeline gpu   # GPU前処理
  uv run infer-onnx model.onnx --pipeline fast   # CPU最適化 (Plan A)
  uv run infer-onnx model.onnx --pipeline current # 従来 (PIL)
        """,
    )

    parser.add_argument("model_path", help="ONNXモデルファイルパス")
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
        help="結果出力ディレクトリ（省略時はモデルパスから自動決定）",
    )
    parser.add_argument(
        "--pipeline",
        choices=PIPELINE_CHOICES,
        default="auto",
        help="前処理パイプライン: auto(デフォルト), current(PIL), fast(CPU最適化), gpu(GPU前処理)",
    )
    parser.add_argument(
        "--benchmark-json",
        action="store_true",
        help="ベンチマーク結果を benchmark_result.json として出力する",
    )
    parser.add_argument(
        "--benchmark-env-name",
        default=None,
        help="ベンチマーク結果の環境ラベル（省略時は自動決定）",
    )

    args = parser.parse_args()
    orchestration_service = OnnxInferenceService()

    manager = LoggerManager()
    level = LogLevel.DEBUG if args.debug else LogLevel.INFO
    manager.set_default_level(level)
    manager.set_logger_level(__name__, level)

    # パス検証
    model_path = Path(args.model_path)
    validate_model_path(model_path)

    # config自動検出・読み込み
    config = load_config_auto(model_path)

    cli_request = InferenceCliRequest(
        model_path=model_path,
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

    # configからパラメータ取得
    use_gpu = config.get("device", "cpu") == "cuda"
    val_transform = config["val_transform"]

    # パイプライン解決
    pipeline = orchestration_service.resolve_pipeline(args.pipeline, use_gpu)
    runtime_options = orchestration_service.resolve_runtime_options(
        config=config,
        pipeline=pipeline,
        use_gpu=use_gpu,
    )
    batch_size = runtime_options.batch_size
    num_workers = runtime_options.num_workers
    pin_memory = runtime_options.pin_memory

    # データセット作成
    dataset, pipeline, norm_mean, norm_std = _create_dataset_and_params(
        pipeline, data_path, val_transform
    )
    use_gpu_pipeline = runtime_options.use_gpu_pipeline

    logger.debug(f"モデル: {model_path}")
    logger.debug(f"データ: {data_path}")
    logger.debug(f"バッチサイズ: {batch_size}")
    logger.debug(f"ワーカー数: {num_workers}")
    logger.debug(f"GPU使用: {use_gpu}")
    logger.debug(f"パイプライン: {pipeline}")
    logger.debug(f"出力先: {output_dir}")

    # DataLoader作成
    data_loader: DataLoader[Any] = DataLoader(
        dataset,
        batch_size=batch_size,
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

    # ONNX推論クラス作成
    logger.debug("ONNXセッションを作成中...")
    inference = OnnxInference(model_path, use_gpu=use_gpu)

    # CUDA EP 不可時のフォールバック: OnnxInference が内部で use_gpu=False に
    # 切り替えた場合, パイプラインとデータセットを再解決する
    if use_gpu and not inference.use_gpu:
        logger.warning("CUDA ExecutionProviderが利用できません.CPUに切り替えます.")
        use_gpu = False
        pipeline = orchestration_service.resolve_pipeline(args.pipeline, use_gpu)
        runtime_options = orchestration_service.resolve_runtime_options(
            config=config,
            pipeline=pipeline,
            use_gpu=use_gpu,
        )
        batch_size = runtime_options.batch_size
        num_workers = runtime_options.num_workers
        pin_memory = runtime_options.pin_memory
        dataset, pipeline, norm_mean, norm_std = _create_dataset_and_params(
            pipeline, data_path, val_transform
        )
        use_gpu_pipeline = runtime_options.use_gpu_pipeline

        # DataLoaderを再作成
        data_loader = DataLoader(
            dataset,
            batch_size=runtime_options.batch_size,
            shuffle=False,
            num_workers=runtime_options.num_workers,
            pin_memory=runtime_options.pin_memory,
        )

    mean_255 = None
    std_255 = None
    if use_gpu_pipeline:
        assert norm_mean is not None and norm_std is not None
        mean_255, std_255 = create_scaled_normalize_tensors(norm_mean, norm_std)

    # 推論実行（End-to-End計測の開始）
    logger.info("推論を開始します...")

    # 入力サイズの取得 (ONNXの入力形状から)
    input_size = None
    try:
        # get_inputs()[0].shape から [batch, channel, height, width] を抽出
        shape = inference.session.get_inputs()[0].shape
        input_size = orchestration_service.resolve_input_size(shape)
    except Exception:
        pass

    execution_request = ExecutionRequest(
        use_gpu_pipeline=use_gpu_pipeline,
        mean_255=mean_255,
        std_255=std_255,
        warmup_repeats=10,
        skip_measurement_batches=1,
        use_cuda_timing=use_gpu,
        gpu_non_blocking=bool(config.get("gpu_non_blocking", True)),
    )
    logger.debug("ウォームアップ中...")
    execution_service = ExecutionService()
    runtime_adapter = OnnxRuntimeAdapter(inference)
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
    providers = inference.get_providers()

    export_service = ResultExportService(logger)
    export_service.export(
        ResultExportRequest(
            output_dir=output_dir,
            model_path=model_path,
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
            results_filename="onnx_inference_results.csv",
            summary_filename="onnx_inference_summary.txt",
            extra_info={"プロバイダー": providers, "パイプライン": pipeline},
            cm_config=config.get("confusion_matrix_config", None),
        )
    )

    if args.benchmark_json:
        configured_env_name = args.benchmark_env_name or config.get(
            "benchmark_env_name"
        )
        env_name = resolve_env_name(
            use_gpu=use_gpu,
            configured_env_name=(
                str(configured_env_name) if configured_env_name is not None else None
            ),
        )
        benchmark_result = build_onnx_benchmark_result(
            use_gpu=use_gpu,
            pipeline=pipeline,
            model_name=str(config.get("model_name", model_path.stem)),
            batch_size=batch_size,
            gpu_non_blocking=execution_request.gpu_non_blocking,
            pin_memory=pin_memory,
            input_size=input_size,
            avg_time_per_image=avg_time_per_image,
            avg_total_time_per_image=avg_total_time_per_image,
            num_samples=num_samples,
            total_samples=total_samples,
            warmup_samples=warmup_samples,
            accuracy=(correct / num_samples * 100.0 if num_samples > 0 else 0.0),
            env_name=env_name,
        )
        try:
            benchmark_json_path = write_benchmark_result_json(
                output_dir=output_dir,
                benchmark_result=benchmark_result,
            )
            logger.info(f"ベンチマークJSONを出力しました: {benchmark_json_path.name}")
        except Exception as exc:  # pragma: no cover
            logger.warning(
                f"ベンチマークJSONの保存に失敗しました, error: {exc}",
            )

    logger.info(f"ワークスペース: {output_dir.name}にサマリーファイルを出力しました")


if __name__ == "__main__":
    main()
