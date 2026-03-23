#!/usr/bin/env python3
"""ONNXモデルを使用した推論CLI.

使用例:
    uv run infer-onnx work_dirs/20260118_001/models/model.onnx
    uv run infer-onnx work_dirs/20260118_001/models/model.onnx --data other/val
    uv run infer-onnx work_dirs/20260118_001/models/model.onnx --pipeline gpu
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
    build_onnx_benchmark_result,
    export_benchmark_json,
    resolve_benchmark_env_name,
)
from pochitrain.inference.services.onnx_inference_service import OnnxInferenceService
from pochitrain.utils import (
    load_config_auto,
    validate_model_path,
)

PIPELINE_CHOICES = ("auto", "current", "fast", "gpu")


def _export_benchmark_if_needed(
    args: argparse.Namespace,
    config: dict,
    result: InferencePipelineResult,
    model_path: Path,
    use_gpu: bool,
    logger: object,
) -> None:
    """ONNX ベンチマーク JSON を条件付きでエクスポートする."""
    if not args.benchmark_json:
        return
    env_name = resolve_benchmark_env_name(
        use_gpu=use_gpu,
        cli_env_name=args.benchmark_env_name,
        config_env_name=config.get("benchmark_env_name"),
    )
    benchmark_result = build_onnx_benchmark_result(
        use_gpu=use_gpu,
        pipeline=result.pipeline,
        model_name=str(config.get("model_name", model_path.stem)),
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
        description="ONNXモデルを使用した推論",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  基本（config・データパスはモデルパスから自動検出）
  uv run infer-onnx work_dirs/20260118_001/models/model.onnx

  データパスを上書き
  uv run infer-onnx work_dirs/20260118_001/models/model.onnx --data other/val

  出力先を上書き
  uv run infer-onnx work_dirs/20260118_001/models/model.onnx -o results/

  uv run infer-onnx model.onnx --pipeline gpu   # GPU前処理
  uv run infer-onnx model.onnx --pipeline fast   # CPU最適化 (Plan A)
  uv run infer-onnx model.onnx --pipeline current # 従来 (PIL)
        """,
    )

    parser.add_argument("model_path", help="ONNXモデルファイルパス")
    parser.add_argument("--debug", action="store_true", help="デバッグログを有効化")
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
    logger = setup_logging(debug=args.debug)

    model_path = Path(args.model_path)
    try:
        validate_model_path(model_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    try:
        config = load_config_auto(model_path)
    except (FileNotFoundError, RuntimeError) as e:
        logger.error(str(e))
        sys.exit(1)

    # ONNX 固有: セッション作成と実 GPU 利用可否の判定
    service = OnnxInferenceService()
    requested_use_gpu = config.get("device", "cpu") == "cuda"
    inference, actual_use_gpu = service.create_onnx_session(
        model_path=model_path, use_gpu=requested_use_gpu
    )

    # ONNX 固有: 入力サイズ解決
    input_size = None
    try:
        shape = inference.session.get_inputs()[0].shape
        input_size = service.resolve_input_size(shape)
    except Exception:
        pass

    providers = inference.get_providers()

    try:
        result = run_inference_pipeline(
            service=service,
            logger=logger,
            model_path=model_path,
            config=config,
            inference=inference,
            args=args,
            use_gpu=actual_use_gpu,
            val_transform=config["val_transform"],
            use_cuda_timing=actual_use_gpu,
            input_size=input_size,
            results_filename="onnx_inference_results.csv",
            summary_filename="onnx_inference_summary.txt",
            extra_info={"プロバイダー": providers, "パイプライン": args.pipeline},
        )
    except (ValueError, Exception) as e:
        logger.error(str(e))
        sys.exit(1)

    _export_benchmark_if_needed(
        args, config, result, model_path, actual_use_gpu, logger
    )


if __name__ == "__main__":
    main()
