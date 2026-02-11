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
from pochitrain.inference.pipeline_strategy import (
    create_dataset_and_params as shared_create_dataset_and_params,
)
from pochitrain.inference.services.execution_service import ExecutionService
from pochitrain.inference.types.execution_types import ExecutionRequest
from pochitrain.logging import LoggerManager
from pochitrain.logging.logger_manager import LogLevel
from pochitrain.onnx import OnnxInference
from pochitrain.pochi_dataset import (
    PochiImageDataset,
    create_scaled_normalize_tensors,
)
from pochitrain.utils import (
    get_default_output_base_dir,
    load_config_auto,
    log_inference_result,
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


def _resolve_pipeline(
    requested: str,
    use_gpu: bool,
    val_transform: Any,
) -> str:
    """--pipeline auto を実際のパイプライン名に解決する.

    Args:
        requested: ユーザー指定のパイプライン名
        use_gpu: GPU推論かどうか
        val_transform: configのval_transform

    Returns:
        解決後のパイプライン名 ("current", "fast", "gpu")
    """
    if requested != "auto":
        if requested == "gpu" and not use_gpu:
            logger.warning(
                "gpuパイプラインが指定されましたが, CPU推論のため fastにフォールバックします"
            )
            return "fast"
        return requested

    # auto: GPU推論 → gpu, CPU推論 → fast
    if use_gpu:
        return "gpu"
    return "fast"


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

    args = parser.parse_args()

    manager = LoggerManager()
    level = LogLevel.DEBUG if args.debug else LogLevel.INFO
    manager.set_default_level(level)
    manager.set_logger_level(__name__, level)

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
        base_dir = get_default_output_base_dir(model_path)
        workspace_manager = InferenceWorkspaceManager(str(base_dir))
        output_dir = workspace_manager.create_workspace()

    # configからパラメータ取得
    batch_size = config.get("batch_size", 1)
    num_workers = config.get("num_workers", 0)
    pin_memory = config.get("pin_memory", True)
    use_gpu = config.get("device", "cpu") == "cuda"
    val_transform = config["val_transform"]

    # パイプライン解決
    pipeline = _resolve_pipeline(args.pipeline, use_gpu, val_transform)

    # データセット作成
    dataset, pipeline, norm_mean, norm_std = _create_dataset_and_params(
        pipeline, data_path, val_transform
    )
    use_gpu_pipeline = pipeline == "gpu"

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
        pipeline = _resolve_pipeline(args.pipeline, use_gpu, val_transform)
        dataset, pipeline, norm_mean, norm_std = _create_dataset_and_params(
            pipeline, data_path, val_transform
        )
        use_gpu_pipeline = pipeline == "gpu"

        # DataLoaderを再作成
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
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
        if len(shape) == 4:
            c = shape[1] if isinstance(shape[1], int) else 3
            h = shape[2] if isinstance(shape[2], int) else None
            w = shape[3] if isinstance(shape[3], int) else None
            if h and w:
                input_size = (c, h, w)
    except Exception:
        pass

    execution_request = ExecutionRequest(
        use_gpu_pipeline=use_gpu_pipeline,
        mean_255=mean_255,
        std_255=std_255,
        warmup_repeats=10,
        skip_measurement_batches=1,
        use_cuda_timing=use_gpu,
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
        avg_total_time_per_image=avg_total_time_per_image,
        input_size=input_size,
        filename="onnx_inference_summary.txt",
        extra_info={"実行プロバイダー": providers, "パイプライン": pipeline},
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
