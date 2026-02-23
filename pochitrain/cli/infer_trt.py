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

from pochitrain.inference.benchmark import (
    build_trt_benchmark_result,
    resolve_env_name,
    write_benchmark_result_json,
)
from pochitrain.inference.services.trt_inference_service import TensorRTInferenceService
from pochitrain.inference.types.orchestration_types import InferenceCliRequest
from pochitrain.logging import LoggerManager
from pochitrain.logging.logger_manager import LogLevel
from pochitrain.utils import (
    load_config_auto,
    validate_model_path,
)

logger: logging.Logger = LoggerManager().get_logger(__name__)

PIPELINE_CHOICES = ("auto", "current", "fast", "gpu")


def main() -> None:
    """メイン関数."""
    parser = argparse.ArgumentParser(
        description="TensorRTエンジンを使用した高速推論",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  基本（config・データパスはエンジンパスから自動検出）
  uv run infer-trt work_dirs/20260118_001/models/model.engine

  データパスを上書き
  uv run infer-trt work_dirs/20260118_001/models/model.engine --data other/val

  出力先を上書き
  uv run infer-trt work_dirs/20260118_001/models/model.engine -o results/

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
    orchestration_service = TensorRTInferenceService()

    manager = LoggerManager()
    level = LogLevel.DEBUG if args.debug else LogLevel.INFO
    manager.set_default_level(level)
    manager.set_logger_level(__name__, level)

    engine_path = Path(args.engine_path)
    validate_model_path(engine_path)

    try:
        inference = orchestration_service.create_trt_inference(engine_path)
    except ImportError:
        logger.error(
            "TensorRTがインストールされていません. "
            "TensorRT SDKをインストールしてください."
        )
        sys.exit(1)

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

    val_transform = orchestration_service.resolve_val_transform(config, inference)

    pipeline = orchestration_service.resolve_pipeline(args.pipeline, use_gpu=True)
    runtime_options = orchestration_service.resolve_runtime_options(config, pipeline)
    data_loader, dataset, pipeline, norm_mean, norm_std = (
        orchestration_service.create_dataloader(
            config=config,
            data_path=data_path,
            val_transform=val_transform,
            pipeline=pipeline,
            runtime_options=runtime_options,
        )
    )

    logger.debug(f"エンジン: {engine_path}")
    logger.debug(f"データ: {data_path}")
    logger.debug(f"ワーカー数: {runtime_options.num_workers}")
    logger.debug(f"パイプライン: {pipeline}")
    logger.debug(f"出力先: {output_dir}")

    logger.info("推論を開始します...")

    input_size = None
    try:
        shape = inference.input_shape
        input_size = orchestration_service.resolve_input_size(shape)
    except Exception:
        pass

    runtime_adapter = orchestration_service.create_runtime_adapter(inference)
    runtime_request = orchestration_service.build_runtime_execution_request(
        data_loader=data_loader,
        runtime_adapter=runtime_adapter,
        use_gpu_pipeline=runtime_options.use_gpu_pipeline,
        norm_mean=norm_mean,
        norm_std=norm_std,
        use_cuda_timing=True,
        gpu_non_blocking=bool(config.get("gpu_non_blocking", True)),
    )
    run_result = orchestration_service.run(runtime_request)
    logger.info("推論完了")

    orchestration_service.aggregate_and_export(
        workspace_dir=output_dir,
        model_path=engine_path,
        data_path=data_path,
        dataset=dataset,
        run_result=run_result,
        input_size=input_size,
        model_info=None,
        cm_config=config.get("confusion_matrix_config", None),
        results_filename="tensorrt_inference_results.csv",
        summary_filename="tensorrt_inference_summary.txt",
        extra_info={"パイプライン": pipeline},
    )

    if args.benchmark_json:
        configured_env_name = args.benchmark_env_name or config.get(
            "benchmark_env_name"
        )
        env_name = resolve_env_name(
            use_gpu=True,
            configured_env_name=(
                str(configured_env_name) if configured_env_name is not None else None
            ),
        )
        benchmark_result = build_trt_benchmark_result(
            engine_path=engine_path,
            pipeline=pipeline,
            model_name=str(config.get("model_name", engine_path.stem)),
            batch_size=runtime_options.batch_size,
            gpu_non_blocking=runtime_request.execution_request.gpu_non_blocking,
            pin_memory=runtime_options.pin_memory,
            input_size=input_size,
            avg_time_per_image=run_result.avg_time_per_image,
            avg_total_time_per_image=run_result.avg_total_time_per_image,
            num_samples=run_result.num_samples,
            total_samples=run_result.total_samples,
            warmup_samples=run_result.warmup_samples,
            accuracy=run_result.accuracy_percent,
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
