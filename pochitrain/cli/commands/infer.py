"""infer サブコマンドの実装."""

import argparse
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from pochitrain import PochiConfig
from pochitrain.cli.cli_commons import setup_logging
from pochitrain.inference.benchmark import (
    build_pytorch_benchmark_result,
    export_benchmark_json,
    resolve_benchmark_env_name,
)
from pochitrain.inference.services import PyTorchInferenceService
from pochitrain.inference.types.orchestration_types import (
    InferenceCliRequest,
)
from pochitrain.utils import (
    ConfigLoader,
    load_config_auto,
    validate_model_path,
)


def infer_command(args: argparse.Namespace) -> None:
    """推論サブコマンドの実行."""
    logger = setup_logging(debug=args.debug)
    logger.debug("=== pochitrain 推論モード ===")

    model_path = Path(args.model_path)
    try:
        validate_model_path(model_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    logger.debug(f"使用するモデル: {model_path}")

    if args.config_path:
        config_path = Path(args.config_path)
        try:
            config = ConfigLoader.load_config(str(config_path))
            logger.debug(f"設定ファイルを読み込み: {config_path}")
        except Exception as e:
            logger.error(f"設定ファイル読み込みエラー: {e}")
            return
    else:
        try:
            config = load_config_auto(model_path)
        except (FileNotFoundError, RuntimeError) as e:
            logger.error(str(e))
            return

    try:
        pochi_config = PochiConfig.from_dict(config)
    except ValidationError as e:
        logger.error(f"設定にエラーがあります:\n{e}")
        return

    service = PyTorchInferenceService(logger)

    requested_pipeline = str(getattr(args, "pipeline", "current"))
    cli_request = InferenceCliRequest(
        model_path=model_path,
        data_path=Path(args.data) if args.data else None,
        output_dir=Path(args.output) if args.output else None,
        requested_pipeline=requested_pipeline,
    )
    try:
        resolved_paths = service.resolve_paths(cli_request, config)
    except ValueError as e:
        logger.error(str(e))
        return
    except Exception as e:
        logger.error(f"パス解決エラー: {e}")
        return

    data_path = resolved_paths.data_path
    workspace_dir = resolved_paths.output_dir

    use_gpu = pochi_config.device == "cuda"
    pipeline = service.resolve_pipeline(
        cli_request.requested_pipeline,
        use_gpu=use_gpu,
    )
    runtime_options = service.resolve_runtime_options(
        config=config,
        pipeline=pipeline,
        use_gpu=use_gpu,
    )

    try:
        predictor = service.create_predictor(pochi_config, model_path)
    except Exception as e:
        logger.error(f"推論器作成エラー: {e}")
        return

    logger.debug("データローダーを作成しています...")
    try:
        (
            val_loader,
            val_dataset,
            pipeline,
            norm_mean,
            norm_std,
        ) = service.create_dataloader(
            config,
            data_path,
            pochi_config.val_transform,
            pipeline,
            runtime_options,
        )
    except Exception as e:
        logger.error(f"データローダー作成エラー: {e}")
        return

    input_size = service.detect_input_size(pochi_config, val_dataset)

    try:
        runtime_adapter = service.create_runtime_adapter(predictor)
        runtime_request = service.build_runtime_execution_request(
            data_loader=val_loader,
            runtime_adapter=runtime_adapter,
            use_gpu_pipeline=pipeline == "gpu",
            norm_mean=norm_mean,
            norm_std=norm_std,
            use_cuda_timing=runtime_adapter.use_cuda_timing,
            gpu_non_blocking=bool(config.get("gpu_non_blocking", True)),
        )
        run_result = service.run(
            runtime_request,
        )
    except Exception as e:
        logger.error(f"推論実行エラー: {e}")
        return

    try:
        cm_config = (
            cast(dict[str, Any], pochi_config.confusion_matrix_config.model_dump())
            if pochi_config.confusion_matrix_config is not None
            else None
        )

        service.aggregate_and_export(
            workspace_dir=workspace_dir,
            model_path=model_path,
            data_path=data_path,
            dataset=val_dataset,
            run_result=run_result,
            input_size=input_size,
            model_info=predictor.get_model_info(),
            cm_config=cm_config,
            results_filename="pytorch_inference_results.csv",
            summary_filename="pytorch_inference_summary.txt",
        )

        if bool(getattr(args, "benchmark_json", False)):
            env_name = resolve_benchmark_env_name(
                use_gpu=use_gpu,
                cli_env_name=getattr(args, "benchmark_env_name", None),
                config_env_name=config.get("benchmark_env_name"),
            )
            benchmark_result = build_pytorch_benchmark_result(
                use_gpu=use_gpu,
                pipeline=pipeline,
                model_name=str(config.get("model_name", model_path.stem)),
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
            export_benchmark_json(workspace_dir, benchmark_result, logger)

        logger.info("推論完了")
        logger.info(
            f"ワークスペース: {workspace_dir.name}へサマリーファイルを出力しました"
        )
    except Exception as e:
        logger.error(f"CSV出力エラー: {e}")
        return
