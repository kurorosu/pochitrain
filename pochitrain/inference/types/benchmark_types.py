"""ベンチマーク結果JSONの型定義とスキーマ定義."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

BENCHMARK_RESULT_SCHEMA_VERSION = "1.0.0"
BENCHMARK_RESULT_FILENAME = "benchmark_result.json"


@dataclass(frozen=True)
class BenchmarkOptions:
    """ベンチ実行時のオプション."""

    gpu_non_blocking: bool
    pin_memory: bool
    batch_size: int
    image_size: Optional[Tuple[int, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式へ変換する.

        Returns:
            JSON出力可能な辞書.
        """
        return {
            "gpu_non_blocking": self.gpu_non_blocking,
            "pin_memory": self.pin_memory,
            "batch_size": self.batch_size,
            "image_size": (
                list(self.image_size) if self.image_size is not None else None
            ),
        }


@dataclass(frozen=True)
class BenchmarkMetrics:
    """ベンチ計測値."""

    avg_inference_ms: float
    avg_e2e_ms: float
    throughput_inference_ips: Optional[float] = None
    throughput_e2e_ips: Optional[float] = None
    accuracy_percent: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式へ変換する.

        Returns:
            JSON出力可能な辞書.
        """
        return {
            "avg_inference_ms": self.avg_inference_ms,
            "avg_e2e_ms": self.avg_e2e_ms,
            "throughput_inference_ips": self.throughput_inference_ips,
            "throughput_e2e_ips": self.throughput_e2e_ips,
            "accuracy_percent": self.accuracy_percent,
        }


@dataclass(frozen=True)
class BenchmarkSamples:
    """サンプル数関連の計測値."""

    num_samples: int
    measured_samples: int
    warmup_samples: int

    def to_dict(self) -> Dict[str, int]:
        """辞書形式へ変換する.

        Returns:
            JSON出力可能な辞書.
        """
        return {
            "num_samples": self.num_samples,
            "measured_samples": self.measured_samples,
            "warmup_samples": self.warmup_samples,
        }


@dataclass(frozen=True)
class BenchmarkResult:
    """`benchmark_result.json` のルート型."""

    timestamp_jst: str
    env_name: str
    runtime: str
    model_name: str
    pipeline: str
    device: str
    options: BenchmarkOptions
    metrics: BenchmarkMetrics
    samples: BenchmarkSamples
    precision: Optional[str] = None
    suite_name: Optional[str] = None
    repeat_index: Optional[int] = None
    schema_version: str = BENCHMARK_RESULT_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式へ変換する.

        Returns:
            JSON出力可能な辞書.
        """
        return {
            "schema_version": self.schema_version,
            "timestamp_jst": self.timestamp_jst,
            "env_name": self.env_name,
            "suite_name": self.suite_name,
            "repeat_index": self.repeat_index,
            "runtime": self.runtime,
            "precision": self.precision,
            "model_name": self.model_name,
            "pipeline": self.pipeline,
            "device": self.device,
            "options": self.options.to_dict(),
            "metrics": self.metrics.to_dict(),
            "samples": self.samples.to_dict(),
        }


def benchmark_result_json_schema() -> Dict[str, Any]:
    """`benchmark_result.json` のJSON Schemaを返す.

    Returns:
        Draft 2020-12 に準拠したJSON Schema辞書.
    """
    nullable_string_schema = {
        "anyOf": [
            {"type": "string"},
            {"type": "null"},
        ]
    }
    nullable_number_schema = {
        "anyOf": [
            {"type": "number"},
            {"type": "null"},
        ]
    }

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "benchmark_result",
        "type": "object",
        "additionalProperties": False,
        "required": [
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
        ],
        "properties": {
            "schema_version": {
                "type": "string",
                "const": BENCHMARK_RESULT_SCHEMA_VERSION,
            },
            "timestamp_jst": {
                "type": "string",
                "pattern": r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$",
            },
            "env_name": {"type": "string"},
            "suite_name": nullable_string_schema,
            "repeat_index": {
                "anyOf": [
                    {"type": "integer", "minimum": 1},
                    {"type": "null"},
                ]
            },
            "runtime": {"type": "string"},
            "precision": nullable_string_schema,
            "model_name": {"type": "string"},
            "pipeline": {"type": "string"},
            "device": {"type": "string"},
            "options": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "gpu_non_blocking",
                    "pin_memory",
                    "batch_size",
                    "image_size",
                ],
                "properties": {
                    "gpu_non_blocking": {"type": "boolean"},
                    "pin_memory": {"type": "boolean"},
                    "batch_size": {"type": "integer", "minimum": 1},
                    "image_size": {
                        "anyOf": [
                            {
                                "type": "array",
                                "items": {"type": "integer", "minimum": 1},
                                "minItems": 2,
                                "maxItems": 2,
                            },
                            {"type": "null"},
                        ]
                    },
                },
            },
            "metrics": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "avg_inference_ms",
                    "avg_e2e_ms",
                    "throughput_inference_ips",
                    "throughput_e2e_ips",
                    "accuracy_percent",
                ],
                "properties": {
                    "avg_inference_ms": {"type": "number", "minimum": 0},
                    "avg_e2e_ms": {"type": "number", "minimum": 0},
                    "throughput_inference_ips": nullable_number_schema,
                    "throughput_e2e_ips": nullable_number_schema,
                    "accuracy_percent": nullable_number_schema,
                },
            },
            "samples": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "num_samples",
                    "measured_samples",
                    "warmup_samples",
                ],
                "properties": {
                    "num_samples": {"type": "integer", "minimum": 0},
                    "measured_samples": {"type": "integer", "minimum": 0},
                    "warmup_samples": {"type": "integer", "minimum": 0},
                },
            },
        },
    }
