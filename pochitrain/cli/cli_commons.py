"""CLI 共通ユーティリティ.

全サブコマンドで共有されるロギング設定やシグナルハンドラーを提供する.
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Any, Optional

from pochitrain.inference.benchmark import export_benchmark_json
from pochitrain.inference.types.orchestration_types import InferenceCliRequest
from pochitrain.logging import LoggerManager
from pochitrain.logging.logger_manager import LogLevel

training_interrupted = False


def create_signal_handler(debug: bool = False) -> Any:
    """デバッグフラグを保持するシグナルハンドラーを生成する.

    Args:
        debug (bool): デバッグモードが有効かどうか

    Returns:
        シグナルハンドラー関数
    """

    def signal_handler(signum: int, frame: Optional[FrameType]) -> None:
        """Ctrl+Cのシグナルハンドラー."""
        global training_interrupted
        training_interrupted = True

        logger = setup_logging(debug=debug)
        logger.warning("訓練を安全に停止しています... (Ctrl+Cが検出されました)")
        logger.warning("現在のエポックが完了次第、訓練を終了します。")

    return signal_handler


def setup_logging(
    logger_name: str = "pochitrain", debug: bool = False
) -> logging.Logger:
    """ログ設定の初期化.

    Args:
        logger_name (str): ロガー名
        debug (bool): デバッグモードが有効かどうか

    Returns:
        logger: 設定済みロガー
    """
    logger_manager = LoggerManager()
    level = LogLevel.DEBUG if debug else LogLevel.INFO
    logger_manager.set_default_level(level)
    for existing_name in logger_manager.get_available_loggers():
        logger_manager.set_logger_level(existing_name, level)
    return logger_manager.get_logger(logger_name, level=level)


@dataclass
class InferencePipelineResult:
    """run_inference_pipeline() の戻り値."""

    run_result: Any
    runtime_request: Any
    runtime_options: Any
    input_size: Optional[tuple[int, int, int]]
    output_dir: Path
    pipeline: str


def run_inference_pipeline(
    *,
    service: Any,
    logger: logging.Logger,
    model_path: Path,
    config: dict[str, Any],
    inference: Any,
    args: argparse.Namespace,
    use_gpu: bool,
    val_transform: Any,
    use_cuda_timing: bool,
    input_size: Optional[tuple[int, int, int]],
    results_filename: str,
    summary_filename: str,
    extra_info: Optional[dict[str, Any]] = None,
) -> InferencePipelineResult:
    """推論 CLI の共通パイプラインを実行する.

    パス解決, データローダー作成, 推論実行, 結果エクスポートを一括処理する.
    ONNX / TensorRT の両 CLI で共有される.

    Args:
        service: IInferenceService 実装.
        logger: ロガー.
        model_path: モデルファイルパス.
        config: 設定辞書.
        inference: ランタイム固有の推論インスタンス.
        args: CLI 引数 (--data, --output, --pipeline を参照).
        use_gpu: GPU 推論かどうか.
        val_transform: 検証用 transform.
        use_cuda_timing: CUDA タイミング計測を使用するか.
        input_size: 入力サイズ (C, H, W). None も可.
        results_filename: 推論結果 CSV ファイル名.
        summary_filename: サマリーテキストファイル名.
        extra_info: サマリーに追加する情報.

    Returns:
        InferencePipelineResult.
    """
    cli_request = InferenceCliRequest(
        model_path=model_path,
        data_path=Path(args.data) if args.data else None,
        output_dir=Path(args.output) if args.output else None,
        requested_pipeline=args.pipeline,
    )
    resolved_paths = service.resolve_paths(cli_request, config)
    data_path = resolved_paths.data_path
    output_dir = resolved_paths.output_dir

    pipeline = service.resolve_pipeline(args.pipeline, use_gpu=use_gpu)
    runtime_options = service.resolve_runtime_options(
        config=config, pipeline=pipeline, use_gpu=use_gpu
    )
    data_loader, dataset, pipeline, norm_mean, norm_std = service.create_dataloader(
        config=config,
        data_path=data_path,
        val_transform=val_transform,
        pipeline=pipeline,
        runtime_options=runtime_options,
    )

    logger.info("推論を開始します...")
    runtime_adapter = service.create_runtime_adapter(inference)
    runtime_request = service.build_runtime_execution_request(
        data_loader=data_loader,
        runtime_adapter=runtime_adapter,
        use_gpu_pipeline=runtime_options.use_gpu_pipeline,
        norm_mean=norm_mean,
        norm_std=norm_std,
        use_cuda_timing=use_cuda_timing,
        gpu_non_blocking=bool(config.get("gpu_non_blocking", True)),
    )
    run_result = service.run(runtime_request)
    logger.info("推論完了")

    service.aggregate_and_export(
        workspace_dir=output_dir,
        model_path=model_path,
        data_path=data_path,
        dataset=dataset,
        run_result=run_result,
        input_size=input_size,
        model_info=None,
        cm_config=config.get("confusion_matrix_config", None),
        results_filename=results_filename,
        summary_filename=summary_filename,
        extra_info=extra_info,
    )

    logger.info(f"ワークスペース: {output_dir.name}にサマリーファイルを出力しました")

    return InferencePipelineResult(
        run_result=run_result,
        runtime_request=runtime_request,
        runtime_options=runtime_options,
        input_size=input_size,
        output_dir=output_dir,
        pipeline=pipeline,
    )
