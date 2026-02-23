"""PyTorch モデル推論のオーケストレーションサービス."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from torch.utils.data import DataLoader

from pochitrain.config import PochiConfig
from pochitrain.logging import LoggerManager
from pochitrain.pochi_dataset import PochiImageDataset
from pochitrain.pochi_predictor import PochiPredictor
from pochitrain.utils import (
    get_default_output_base_dir,
    log_inference_result,
    validate_data_path,
)
from pochitrain.utils.directory_manager import InferenceWorkspaceManager

from ..types.orchestration_types import (
    InferenceCliRequest,
    InferenceResolvedPaths,
    InferenceRuntimeOptions,
)
from ..types.result_export_types import ResultExportRequest
from .result_export_service import ResultExportService


class PyTorchInferenceService:
    """PyTorch モデル推論の実行・集約・エクスポートを担うサービス.

    CLI から推論ビジネスロジックを分離し, 単体テストを可能にする.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """サービスを初期化する.

        Args:
            logger: ロガーインスタンス. 未指定時はモジュールロガーを利用する.
        """
        self.logger = logger or LoggerManager().get_logger(__name__)

    def resolve_paths(
        self,
        request: InferenceCliRequest,
        config: Dict[str, Any],
    ) -> InferenceResolvedPaths:
        """データパスと出力先を解決する.

        Args:
            request: CLI入力を表すリクエスト.
            config: 設定値辞書.

        Returns:
            解決済みパス情報.

        Raises:
            ValueError: データパスが解決できない場合.
        """
        if request.data_path is not None:
            data_path = request.data_path
        elif "val_data_root" in config:
            data_path = Path(config["val_data_root"])
        else:
            raise ValueError(
                "--data を指定するか, configにval_data_rootを設定してください"
            )

        validate_data_path(data_path)

        if request.output_dir is not None:
            output_dir = request.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            base_dir = get_default_output_base_dir(request.model_path)
            workspace_manager = InferenceWorkspaceManager(str(base_dir))
            output_dir = workspace_manager.create_workspace()

        return InferenceResolvedPaths(
            model_path=request.model_path,
            data_path=data_path,
            output_dir=output_dir,
        )

    def resolve_pipeline(self, requested: str) -> str:
        """PyTorch推論で利用するパイプラインを解決する.

        Args:
            requested: 要求されたパイプライン名.

        Returns:
            解決後のパイプライン名.
        """
        if requested == "auto":
            return "current"
        return "current"

    def resolve_runtime_options(
        self,
        config: Dict[str, Any],
        pipeline: str,
        use_gpu: bool,
    ) -> InferenceRuntimeOptions:
        """PyTorch推論時の実行オプションを解決する.

        Args:
            config: 設定値辞書.
            pipeline: 解決済みパイプライン名.
            use_gpu: GPU推論を使うかどうか.

        Returns:
            実行時オプション.
        """
        return InferenceRuntimeOptions(
            pipeline=pipeline,
            batch_size=int(config.get("batch_size", 1)),
            num_workers=int(config.get("num_workers", 0)),
            pin_memory=bool(config.get("pin_memory", True)),
            use_gpu=use_gpu,
            use_gpu_pipeline=False,
        )

    def create_predictor(self, config: PochiConfig, model_path: Path) -> PochiPredictor:
        """推論器を生成する.

        Args:
            config: アプリケーション設定.
            model_path: 学習済みモデルのパス.

        Returns:
            初期化済みの推論器.
        """
        return PochiPredictor.from_config(config, str(model_path))

    def create_dataloader(
        self, config: PochiConfig, data_path: Path
    ) -> Tuple[DataLoader[Any], PochiImageDataset]:
        """推論用 DataLoader とデータセットを生成する.

        Args:
            config: アプリケーション設定.
            data_path: 推論データのディレクトリパス.

        Returns:
            (DataLoader, PochiImageDataset) のタプル.
        """
        dataset = PochiImageDataset(str(data_path), transform=config.val_transform)
        loader: DataLoader[Any] = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )

        self.logger.debug("使用されたTransform (設定ファイルから):")
        for i, transform in enumerate(config.val_transform.transforms):
            self.logger.debug(f"   {i + 1}. {transform}")

        return loader, dataset

    def detect_input_size(
        self, config: PochiConfig, dataset: PochiImageDataset
    ) -> Optional[Tuple[int, int, int]]:
        """Transform またはデータセットから入力サイズを推定する.

        Args:
            config: アプリケーション設定.
            dataset: 推論データセット.

        Returns:
            (C, H, W) のタプル. 取得できない場合は None.
        """
        try:
            from torchvision.transforms import CenterCrop, RandomResizedCrop, Resize

            for t in config.val_transform.transforms:
                if isinstance(t, (Resize, CenterCrop, RandomResizedCrop)):
                    size = getattr(t, "size", None)
                    if size:
                        if isinstance(size, int):
                            return (3, size, size)
                        elif isinstance(size, (list, tuple)):
                            return (3, size[0], size[1])
                        break

            if len(dataset) > 0:
                sample_img, _ = dataset[0]
                if hasattr(sample_img, "shape") and len(sample_img.shape) == 3:
                    s = sample_img.shape
                    return (int(s[0]), int(s[1]), int(s[2]))
        except Exception:
            pass

        return None

    def run_inference(
        self, predictor: PochiPredictor, val_loader: DataLoader[Any]
    ) -> Tuple[List[int], List[float], Dict[str, Any], float]:
        """推論を実行し, 結果と計測情報を返す.

        Args:
            predictor: 推論器.
            val_loader: 推論データローダー.

        Returns:
            (predicted_labels, confidence_scores, metrics, e2e_total_time_ms) のタプル.
        """
        self.logger.info("推論を開始します...")

        e2e_start_time = time.perf_counter()
        predictions, confidences, metrics = predictor.predict(val_loader)
        e2e_total_time_ms = (time.perf_counter() - e2e_start_time) * 1000

        predicted_labels: List[int] = predictions.tolist()
        confidence_scores: List[float] = confidences.tolist()

        return predicted_labels, confidence_scores, metrics, e2e_total_time_ms

    def aggregate_and_export(
        self,
        *,
        workspace_dir: Path,
        model_path: Path,
        data_path: Path,
        dataset: PochiImageDataset,
        predicted_labels: List[int],
        confidence_scores: List[float],
        metrics: Dict[str, Any],
        e2e_total_time_ms: float,
        input_size: Optional[Tuple[int, int, int]],
        model_info: Optional[Dict[str, Any]],
        cm_config: Optional[Dict[str, Any]],
    ) -> None:
        """精度計算・ログ出力・結果エクスポートを実行する.

        Args:
            workspace_dir: 結果出力先ディレクトリ.
            model_path: モデルファイルパス.
            data_path: 推論データパス.
            dataset: 推論データセット.
            predicted_labels: 予測ラベルリスト.
            confidence_scores: 確信度リスト.
            metrics: 推論メトリクス辞書.
            e2e_total_time_ms: End-to-End 全処理時間 (ms).
            input_size: 入力サイズ (C, H, W).
            model_info: モデル情報辞書.
            cm_config: 混同行列可視化設定.
        """
        image_paths = dataset.get_file_paths()
        true_labels = dataset.labels
        class_names = dataset.get_classes()

        num_samples = len(predicted_labels)
        correct = sum(p == t for p, t in zip(predicted_labels, true_labels))
        avg_total_time_per_image = (
            e2e_total_time_ms / num_samples if num_samples > 0 else 0
        )

        log_inference_result(
            num_samples=num_samples,
            correct=correct,
            avg_time_per_image=metrics["avg_time_per_image"],
            total_samples=int(metrics["total_samples"]),
            warmup_samples=int(metrics["warmup_samples"]),
            avg_total_time_per_image=avg_total_time_per_image,
            input_size=input_size,
        )

        export_service = ResultExportService(self.logger)
        export_service.export(
            ResultExportRequest(
                output_dir=workspace_dir,
                model_path=model_path,
                data_path=data_path,
                image_paths=image_paths,
                predictions=predicted_labels,
                true_labels=true_labels,
                confidences=confidence_scores,
                class_names=class_names,
                num_samples=num_samples,
                correct=correct,
                avg_time_per_image=metrics["avg_time_per_image"],
                total_samples=int(metrics["total_samples"]),
                warmup_samples=int(metrics["warmup_samples"]),
                avg_total_time_per_image=avg_total_time_per_image,
                input_size=input_size,
                results_filename="pytorch_inference_results.csv",
                summary_filename="pytorch_inference_summary.txt",
                model_info=model_info,
                cm_config=cm_config,
            )
        )
