"""benchmark_types のユニットテスト."""

from pochitrain.inference.types.benchmark_types import (
    BENCHMARK_RESULT_SCHEMA_VERSION,
    BenchmarkMetrics,
    BenchmarkOptions,
    BenchmarkResult,
    BenchmarkSamples,
    benchmark_result_json_schema,
)


def test_benchmark_result_to_dict_contains_expected_values():
    """to_dict が仕様どおりのキーと値を返すことを確認する."""
    result = BenchmarkResult(
        timestamp_jst="2026-02-22 21:34:56",
        env_name="Windows-RTX4070Ti",
        runtime="tensorrt",
        precision="int8",
        model_name="resnet18",
        pipeline="gpu",
        device="cuda",
        suite_name="gpu_non_blocking",
        repeat_index=1,
        options=BenchmarkOptions(
            gpu_non_blocking=True,
            pin_memory=True,
            batch_size=1,
            image_size=(512, 512),
        ),
        metrics=BenchmarkMetrics(
            avg_inference_ms=0.29,
            avg_e2e_ms=1.70,
            throughput_inference_ips=3448.3,
            throughput_e2e_ips=588.2,
            accuracy_percent=100.0,
        ),
        samples=BenchmarkSamples(
            num_samples=38,
            measured_samples=37,
            warmup_samples=1,
        ),
    )

    payload = result.to_dict()

    assert payload["schema_version"] == BENCHMARK_RESULT_SCHEMA_VERSION
    assert payload["env_name"] == "Windows-RTX4070Ti"
    assert payload["runtime"] == "tensorrt"
    assert payload["precision"] == "int8"
    assert payload["options"]["image_size"] == [512, 512]
    assert payload["metrics"]["avg_inference_ms"] == 0.29
    assert payload["samples"]["warmup_samples"] == 1


def test_benchmark_result_default_schema_version():
    """schema_version の既定値が定数と一致することを確認する."""
    result = BenchmarkResult(
        timestamp_jst="2026-02-22 21:34:56",
        env_name="Jetson-Orin-Nano",
        runtime="onnx",
        model_name="resnet18",
        pipeline="gpu",
        device="cuda",
        options=BenchmarkOptions(
            gpu_non_blocking=False,
            pin_memory=False,
            batch_size=1,
        ),
        metrics=BenchmarkMetrics(
            avg_inference_ms=20.88,
            avg_e2e_ms=28.08,
        ),
        samples=BenchmarkSamples(
            num_samples=38,
            measured_samples=37,
            warmup_samples=1,
        ),
    )

    assert result.schema_version == BENCHMARK_RESULT_SCHEMA_VERSION


def test_benchmark_json_schema_has_required_top_level_keys():
    """JSON Schema が必要なトップレベル要素を要求することを確認する."""
    schema = benchmark_result_json_schema()
    required = set(schema["required"])

    assert required == {
        "schema_version",
        "timestamp_jst",
        "env_name",
        "runtime",
        "model_name",
        "pipeline",
        "device",
        "options",
        "metrics",
        "samples",
    }

    options_required = set(schema["properties"]["options"]["required"])
    assert options_required == {
        "gpu_non_blocking",
        "pin_memory",
        "batch_size",
        "image_size",
    }
