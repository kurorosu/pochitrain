"""orchestration_types のユニットテスト."""

from pochitrain.inference.types.execution_types import ExecutionResult
from pochitrain.inference.types.orchestration_types import InferenceRunResult


def test_inference_run_result_from_execution_result() -> None:
    """ExecutionResult から共通結果型へ変換できることを検証する."""
    execution_result = ExecutionResult(
        predictions=[1, 0, 1],
        confidences=[0.9, 0.8, 0.7],
        true_labels=[1, 0, 0],
        total_inference_time_ms=12.0,
        total_samples=3,
        warmup_samples=1,
        e2e_total_time_ms=30.0,
    )

    result = InferenceRunResult.from_execution_result(execution_result)

    assert result.num_samples == 3
    assert result.correct == 2
    assert result.avg_time_per_image == 4.0
    assert result.avg_total_time_per_image == 10.0
    assert result.accuracy_percent == (2 / 3) * 100.0


def test_inference_run_result_from_components_with_zero_samples() -> None:
    """測定サンプル数が0でもゼロ除算せず集計できることを検証する."""
    result = InferenceRunResult.from_components(
        predictions=[],
        confidences=[],
        true_labels=[],
        total_inference_time_ms=0.0,
        total_samples=0,
        warmup_samples=0,
        e2e_total_time_ms=0.0,
    )

    assert result.num_samples == 0
    assert result.correct == 0
    assert result.avg_time_per_image == 0.0
    assert result.avg_total_time_per_image == 0.0
    assert result.accuracy_percent == 0.0
