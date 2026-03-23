#!/usr/bin/env python3
"""TensorRTエンジンを使用した推論CLI.

使用例:
    uv run infer-trt work_dirs/20260118_001/models/model.engine
    uv run infer-trt work_dirs/20260118_001/models/model.engine --data other/val
    uv run infer-trt work_dirs/20260118_001/models/model.engine --pipeline gpu
"""

import argparse
import sys
from pathlib import Path

from pochitrain.cli.cli_commons import (
    InferencePipelineResult,
    run_inference_pipeline,
    setup_logging,
)
from pochitrain.inference.benchmark import (
    build_trt_benchmark_result,
    export_benchmark_json,
    resolve_benchmark_env_name,
)
from pochitrain.inference.services.trt_inference_service import TensorRTInferenceService
from pochitrain.utils import (
    load_config_auto,
    validate_model_path,
)

PIPELINE_CHOICES = ("auto", "current", "fast", "gpu")


def _export_benchmark_if_needed(
    args: argparse.Namespace,
    config: dict,
    result: InferencePipelineResult,
    engine_path: Path,
    logger: object,
) -> None:
    """条件付きで TensorRT ベンチマーク JSON をエクスポートする."""
    if not args.benchmark_json:
        return
    env_name = resolve_benchmark_env_name(
        use_gpu=True,
        cli_env_name=args.benchmark_env_name,
        config_env_name=config.get("benchmark_env_name"),
    )
    benchmark_result = build_trt_benchmark_result(
        engine_path=engine_path,
        pipeline=result.pipeline,
        model_name=str(config.get("model_name", engine_path.stem)),
        batch_size=result.runtime_options.batch_size,
        gpu_non_blocking=result.runtime_request.execution_request.gpu_non_blocking,
        pin_memory=result.runtime_options.pin_memory,
        input_size=result.input_size,
        avg_time_per_image=result.run_result.avg_time_per_image,
        avg_total_time_per_image=result.run_result.avg_total_time_per_image,
        num_samples=result.run_result.num_samples,
        total_samples=result.run_result.total_samples,
        warmup_samples=result.run_result.warmup_samples,
        accuracy=result.run_result.accuracy_percent,
        env_name=env_name,
    )
    export_benchmark_json(result.output_dir, benchmark_result, logger)  # type: ignore[arg-type]


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

  uv run infer-trt model.engine --pipeline gpu     # GPU前処理
  uv run infer-trt model.engine --pipeline fast     # CPU最適化 (Plan A)
  uv run infer-trt model.engine --pipeline current  # 従来 (PIL)

前提条件:
  - TensorRT SDKのインストールが必要
  - uv pip install TensorRT-10.x.x.x\\python\\tensorrt-10.x.x-cpXX-win_amd64.whl
        """,
    )

    parser.add_argument("engine_path", help="TensorRTエンジンファイルパス (.engine)")
    parser.add_argument("--debug", action="store_true", help="デバッグログを有効化")
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
    logger = setup_logging(debug=args.debug)

    engine_path = Path(args.engine_path)
    try:
        validate_model_path(engine_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # TRT 固有: TensorRT 推論インスタンス作成 (config 読み込み前に必要)
    service = TensorRTInferenceService()
    try:
        inference = service.create_trt_inference(engine_path)
    except ImportError:
        logger.error(
            "TensorRTがインストールされていません. "
            "TensorRT SDKをインストールしてください."
        )
        sys.exit(1)

    try:
        config = load_config_auto(engine_path)
    except (FileNotFoundError, RuntimeError) as e:
        logger.error(str(e))
        sys.exit(1)

    # TRT 固有: val_transform と入力サイズの解決
    val_transform = service.resolve_val_transform(config, inference)

    input_size = None
    try:
        shape = inference.input_shape
        input_size = service.resolve_input_size(shape)
    except Exception:
        pass

    try:
        result = run_inference_pipeline(
            service=service,
            logger=logger,
            model_path=engine_path,
            config=config,
            inference=inference,
            args=args,
            use_gpu=True,
            val_transform=val_transform,
            use_cuda_timing=True,
            input_size=input_size,
            results_filename="tensorrt_inference_results.csv",
            summary_filename="tensorrt_inference_summary.txt",
            extra_info={"パイプライン": args.pipeline},
        )
    except (ValueError, Exception) as e:
        logger.error(str(e))
        sys.exit(1)

    _export_benchmark_if_needed(args, config, result, engine_path, logger)


if __name__ == "__main__":
    main()
