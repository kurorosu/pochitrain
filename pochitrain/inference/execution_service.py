"""ONNX/TRT共通の推論実行ループを提供するサービス."""

import time
from typing import Any, List

import torch
from torch.utils.data import DataLoader

from pochitrain.inference.execution_types import ExecutionRequest, ExecutionResult
from pochitrain.inference.interfaces import IRuntimeAdapter
from pochitrain.utils import post_process_logits


class ExecutionService:
    """推論のウォームアップ, 計測, 集計を共通化するサービス."""

    def run(
        self,
        data_loader: DataLoader[Any],
        runtime: IRuntimeAdapter,
        request: ExecutionRequest,
    ) -> ExecutionResult:
        """推論を実行し, 予測結果と計測結果を集計して返す.

        Args:
            data_loader: 推論対象のDataLoader.
            runtime: 推論ランタイム差分を吸収するアダプタ.
            request: 実行パラメータ.

        Returns:
            予測結果と計測結果の集計値.
        """
        self._run_warmup(data_loader, runtime, request)

        e2e_start_time = time.perf_counter()

        all_predictions: List[int] = []
        all_confidences: List[float] = []
        all_true_labels: List[int] = []
        total_inference_time_ms = 0.0
        total_samples = 0
        warmup_samples = 0

        start_event = None
        end_event = None
        if request.use_cuda_timing and runtime.use_cuda_timing:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

        for batch_idx, (images, labels) in enumerate(data_loader):
            if batch_idx < request.skip_measurement_batches:
                runtime.set_input(images, request)
                runtime.run_inference()
                logits = runtime.get_output()
                predicted, confidence = post_process_logits(logits)
                warmup_samples += len(images)
            else:
                runtime.set_input(images, request)

                if start_event is not None and end_event is not None:
                    start_event.record()
                    runtime.run_inference()
                    end_event.record()
                    torch.cuda.synchronize()
                    inference_time_ms = start_event.elapsed_time(end_event)
                else:
                    start_time = time.perf_counter()
                    runtime.run_inference()
                    inference_time_ms = (time.perf_counter() - start_time) * 1000

                logits = runtime.get_output()
                predicted, confidence = post_process_logits(logits)
                total_inference_time_ms += inference_time_ms
                total_samples += len(images)

            all_predictions.extend(predicted.tolist())
            all_confidences.extend(confidence.tolist())
            all_true_labels.extend(labels.tolist())

        e2e_total_time_ms = (time.perf_counter() - e2e_start_time) * 1000

        return ExecutionResult(
            predictions=all_predictions,
            confidences=all_confidences,
            true_labels=all_true_labels,
            total_inference_time_ms=total_inference_time_ms,
            total_samples=total_samples,
            warmup_samples=warmup_samples,
            e2e_total_time_ms=e2e_total_time_ms,
        )

    def _run_warmup(
        self,
        data_loader: DataLoader[Any],
        runtime: IRuntimeAdapter,
        request: ExecutionRequest,
    ) -> None:
        """データ先頭サンプルを使ってウォームアップを行う.

        Args:
            data_loader: 推論対象のDataLoader.
            runtime: 推論ランタイム差分を吸収するアダプタ.
            request: 実行パラメータ.
        """
        if request.warmup_repeats <= 0:
            return

        dataset = data_loader.dataset
        if len(dataset) == 0:
            return

        image, _ = dataset[0]
        runtime.warmup(image, request)
