"""ベンチマーク結果型の生成処理."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple

from pochitrain.inference.types.benchmark_types import (
    BenchmarkMetrics,
    BenchmarkOptions,
    BenchmarkResult,
    BenchmarkSamples,
)

JST = timezone(timedelta(hours=9))


def _now_jst_timestamp() -> str:
    """現在時刻を `YYYY-MM-DD HH:MM:SS` 形式で返す."""
    return datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")


def _resolve_image_size(
    input_size: Optional[Tuple[int, int, int]],
) -> Optional[Tuple[int, int]]:
    """入力サイズ (C, H, W) を画像サイズ (H, W) へ変換する."""
    return (input_size[1], input_size[2]) if input_size is not None else None


def _throughput_from_ms(avg_ms: float) -> float:
    """平均時間 (ms/image) から throughput (images/sec) を算出する."""
    return 1000.0 / avg_ms if avg_ms > 0 else 0.0


def _build_benchmark_result(
    *,
    runtime: str,
    precision: Optional[str],
    device: str,
    pipeline: str,
    model_name: str,
    batch_size: int,
    gpu_non_blocking: bool,
    pin_memory: bool,
    input_size: Optional[Tuple[int, int, int]],
    avg_time_per_image: float,
    avg_total_time_per_image: float,
    num_samples: int,
    total_samples: int,
    warmup_samples: int,
    accuracy: float,
    env_name: str,
) -> BenchmarkResult:
    """共通形式のベンチマーク結果を構築する."""
    return BenchmarkResult(
        timestamp_jst=_now_jst_timestamp(),
        env_name=env_name,
        runtime=runtime,
        precision=precision,
        model_name=model_name,
        pipeline=pipeline,
        device=device,
        options=BenchmarkOptions(
            gpu_non_blocking=gpu_non_blocking,
            pin_memory=pin_memory,
            batch_size=batch_size,
            image_size=_resolve_image_size(input_size),
        ),
        metrics=BenchmarkMetrics(
            avg_inference_ms=avg_time_per_image,
            avg_e2e_ms=avg_total_time_per_image,
            throughput_inference_ips=_throughput_from_ms(avg_time_per_image),
            throughput_e2e_ips=_throughput_from_ms(avg_total_time_per_image),
            accuracy_percent=accuracy,
        ),
        samples=BenchmarkSamples(
            num_samples=num_samples,
            measured_samples=total_samples,
            warmup_samples=warmup_samples,
        ),
    )


def build_onnx_benchmark_result(
    *,
    use_gpu: bool,
    pipeline: str,
    model_name: str,
    batch_size: int,
    gpu_non_blocking: bool,
    pin_memory: bool,
    input_size: Optional[Tuple[int, int, int]],
    avg_time_per_image: float,
    avg_total_time_per_image: float,
    num_samples: int,
    total_samples: int,
    warmup_samples: int,
    accuracy: float,
    env_name: str,
) -> BenchmarkResult:
    """ONNX推論の集計結果からベンチ結果型を構築する.

    Args:
        use_gpu: GPU利用有無.
        pipeline: 使用した前処理パイプライン.
        model_name: モデル名.
        batch_size: 実行バッチサイズ.
        gpu_non_blocking: non_blocking設定値.
        pin_memory: pin_memory設定値.
        input_size: 入力サイズ (C, H, W).
        avg_time_per_image: 純粋推論平均時間 (ms/image).
        avg_total_time_per_image: E2E平均時間 (ms/image).
        num_samples: データセット全サンプル数.
        total_samples: 実測サンプル数.
        warmup_samples: ウォームアップ除外サンプル数.
        accuracy: 精度(%).
        env_name: 環境ラベル.

    Returns:
        生成したベンチマーク結果.
    """
    return _build_benchmark_result(
        runtime="onnx",
        precision="fp32",
        device="cuda" if use_gpu else "cpu",
        pipeline=pipeline,
        model_name=model_name,
        batch_size=batch_size,
        gpu_non_blocking=gpu_non_blocking,
        pin_memory=pin_memory,
        input_size=input_size,
        avg_time_per_image=avg_time_per_image,
        avg_total_time_per_image=avg_total_time_per_image,
        num_samples=num_samples,
        total_samples=total_samples,
        warmup_samples=warmup_samples,
        accuracy=accuracy,
        env_name=env_name,
    )


def _resolve_trt_precision(engine_path: Path) -> Optional[str]:
    """エンジンファイル名から推定できる精度を返す.

    Args:
        engine_path: TensorRTエンジンファイル.

    Returns:
        推定精度文字列. 推定不能な場合はNone.
    """
    lowered = engine_path.name.lower()
    if "int8" in lowered:
        return "int8"
    if "fp16" in lowered:
        return "fp16"
    if "fp32" in lowered:
        return "fp32"
    return None


def build_trt_benchmark_result(
    *,
    engine_path: Path,
    pipeline: str,
    model_name: str,
    batch_size: int,
    gpu_non_blocking: bool,
    pin_memory: bool,
    input_size: Optional[Tuple[int, int, int]],
    avg_time_per_image: float,
    avg_total_time_per_image: float,
    num_samples: int,
    total_samples: int,
    warmup_samples: int,
    accuracy: float,
    env_name: str,
) -> BenchmarkResult:
    """TensorRT推論の集計結果からベンチ結果型を構築する.

    Args:
        engine_path: TensorRTエンジンファイル.
        pipeline: 使用した前処理パイプライン.
        model_name: モデル名.
        batch_size: 実行バッチサイズ.
        gpu_non_blocking: non_blocking設定値.
        pin_memory: pin_memory設定値.
        input_size: 入力サイズ (C, H, W).
        avg_time_per_image: 純粋推論平均時間 (ms/image).
        avg_total_time_per_image: E2E平均時間 (ms/image).
        num_samples: データセット全サンプル数.
        total_samples: 実測サンプル数.
        warmup_samples: ウォームアップ除外サンプル数.
        accuracy: 精度(%).
        env_name: 環境ラベル.

    Returns:
        生成したベンチマーク結果.
    """
    return _build_benchmark_result(
        runtime="tensorrt",
        precision=_resolve_trt_precision(engine_path),
        device="cuda",
        pipeline=pipeline,
        model_name=model_name,
        batch_size=batch_size,
        gpu_non_blocking=gpu_non_blocking,
        pin_memory=pin_memory,
        input_size=input_size,
        avg_time_per_image=avg_time_per_image,
        avg_total_time_per_image=avg_total_time_per_image,
        num_samples=num_samples,
        total_samples=total_samples,
        warmup_samples=warmup_samples,
        accuracy=accuracy,
        env_name=env_name,
    )


def build_pytorch_benchmark_result(
    *,
    use_gpu: bool,
    pipeline: str,
    model_name: str,
    batch_size: int,
    gpu_non_blocking: bool,
    pin_memory: bool,
    input_size: Optional[Tuple[int, int, int]],
    avg_time_per_image: float,
    avg_total_time_per_image: float,
    num_samples: int,
    total_samples: int,
    warmup_samples: int,
    accuracy: float,
    env_name: str,
) -> BenchmarkResult:
    """PyTorch推論の集計結果からベンチ結果型を構築する.

    Args:
        use_gpu: GPU利用有無.
        pipeline: 使用した前処理パイプライン.
        model_name: モデル名.
        batch_size: 実行バッチサイズ.
        gpu_non_blocking: non_blocking設定値.
        pin_memory: pin_memory設定値.
        input_size: 入力サイズ (C, H, W).
        avg_time_per_image: 純粋推論平均時間 (ms/image).
        avg_total_time_per_image: E2E平均時間 (ms/image).
        num_samples: データセット全サンプル数.
        total_samples: 実測サンプル数.
        warmup_samples: ウォームアップ除外サンプル数.
        accuracy: 精度(%).
        env_name: 環境ラベル.

    Returns:
        生成したベンチマーク結果.
    """
    return _build_benchmark_result(
        runtime="pytorch",
        precision="fp32",
        device="cuda" if use_gpu else "cpu",
        pipeline=pipeline,
        model_name=model_name,
        batch_size=batch_size,
        gpu_non_blocking=gpu_non_blocking,
        pin_memory=pin_memory,
        input_size=input_size,
        avg_time_per_image=avg_time_per_image,
        avg_total_time_per_image=avg_total_time_per_image,
        num_samples=num_samples,
        total_samples=total_samples,
        warmup_samples=warmup_samples,
        accuracy=accuracy,
        env_name=env_name,
    )
