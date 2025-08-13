"""
pochitrain.pochi_predictor: 推論機能のメインモジュール.

学習済みモデルを読み込んで推論を実行する機能を提供します。
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from .pochi_dataset import PochiImageDataset, get_basic_transforms
from .pochi_trainer import PochiTrainer
from .utils.directory_manager import InferenceWorkspaceManager


class PochiPredictor(PochiTrainer):
    """
    推論専用のPochiトレーナー拡張クラス.

    PochiTrainerを継承し、推論に特化した機能を追加します。

    Args:
        model_name (str): モデル名 ('resnet18', 'resnet34', 'resnet50')
        num_classes (int): 分類クラス数
        device (str): デバイス ('cuda' or 'cpu')
        model_path (str): 学習済みモデルのパス
        work_dir (str, optional): 作業ディレクトリ
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        device: str,
        model_path: str,
        work_dir: str = "inference_results",
    ):
        """PochiPredictorを初期化."""
        # 親クラスを初期化（pretrainedはFalseにして後でロード）
        super().__init__(
            model_name=model_name,
            num_classes=num_classes,
            device=device,
            pretrained=False,
            work_dir="temp_workspace",  # 一時的なワークスペース
        )

        # 推論専用ワークスペースマネージャーで上書き
        self.inference_workspace_manager = InferenceWorkspaceManager(work_dir)
        self.inference_workspace = self.inference_workspace_manager.create_workspace()

        self.model_path = Path(model_path)
        self._load_trained_model()

    def _load_trained_model(self) -> None:
        """学習済みモデルを読み込み."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"モデルファイルが見つかりません: {self.model_path}"
            )

        try:
            # チェックポイントの読み込み
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # モデルの状態辞書を読み込み
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()  # 推論モードに設定

            # メタ情報の取得
            if "best_accuracy" in checkpoint:
                self.best_accuracy = checkpoint["best_accuracy"]
                self.logger.info(f"モデルの最高精度: {self.best_accuracy:.2f}%")

            if "epoch" in checkpoint:
                self.epoch = checkpoint["epoch"]
                self.logger.info(f"学習エポック数: {self.epoch}")

            self.logger.info(f"学習済みモデルを読み込み: {self.model_path}")

        except Exception as e:
            raise RuntimeError(f"モデルの読み込みに失敗しました: {e}")

    def predict_with_paths(
        self,
        val_data_root: str,
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 224,
    ) -> Tuple[List[str], List[int], List[int], List[float], List[str]]:
        """
        バリデーションデータに対して推論を実行し、詳細な結果を返す.

        Args:
            val_data_root (str): バリデーションデータのルートディレクトリ
            batch_size (int): バッチサイズ
            num_workers (int): ワーカー数
            image_size (int): 画像サイズ

        Returns:
            Tuple[List[str], List[int], List[int], List[float], List[str]]:
                (画像パス, 推論ラベル, 正解ラベル, 信頼度, クラス名リスト)
        """
        # バリデーション用のデータローダーを作成
        val_transform = get_basic_transforms(image_size=image_size, is_training=False)
        val_dataset = PochiImageDataset(val_data_root, transform=val_transform)

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        self.logger.info(f"推論開始: {len(val_dataset)}枚の画像")

        # 推論実行
        predictions, confidences = self.predict(val_loader)

        # 結果の整理
        image_paths = val_dataset.get_file_paths()
        predicted_labels = predictions.tolist()
        confidence_scores = confidences.tolist()
        true_labels = val_dataset.labels
        class_names = val_dataset.get_classes()

        self.logger.info("推論完了")

        return (
            image_paths,
            predicted_labels,
            true_labels,
            confidence_scores,
            class_names,
        )

    def calculate_accuracy(
        self, predicted_labels: List[int], true_labels: List[int]
    ) -> Dict[str, float]:
        """
        推論結果の精度を計算.

        Args:
            predicted_labels (List[int]): 推論ラベル
            true_labels (List[int]): 正解ラベル

        Returns:
            Dict[str, float]: 精度情報
        """
        total = len(predicted_labels)
        correct = sum(p == t for p, t in zip(predicted_labels, true_labels))
        accuracy = (correct / total) * 100 if total > 0 else 0.0

        accuracy_info = {
            "total_samples": total,
            "correct_predictions": correct,
            "accuracy_percentage": accuracy,
        }

        self.logger.info(f"推論精度: {correct}/{total} ({accuracy:.2f}%)")

        return accuracy_info

    def get_model_info(self) -> Dict[str, Any]:
        """
        モデル情報を取得.

        Returns:
            Dict[str, any]: モデル情報
        """
        return {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "device": str(self.device),
            "model_path": str(self.model_path),
            "best_accuracy": getattr(self, "best_accuracy", None),
            "epoch": getattr(self, "epoch", None),
        }

    def export_results_to_workspace(
        self,
        image_paths: List[str],
        predicted_labels: List[int],
        true_labels: List[int],
        confidence_scores: List[float],
        class_names: List[str],
        results_filename: str = "inference_results.csv",
        summary_filename: str = "inference_summary.csv",
    ) -> Tuple[Path, Path]:
        """
        推論結果をワークスペース内にCSV出力.

        Args:
            image_paths (List[str]): 画像パスのリスト
            predicted_labels (List[int]): 推論ラベルのリスト
            true_labels (List[int]): 正解ラベルのリスト
            confidence_scores (List[float]): 信頼度のリスト
            class_names (List[str]): クラス名のリスト
            results_filename (str): 詳細結果CSVファイル名
            summary_filename (str): サマリーCSVファイル名

        Returns:
            Tuple[Path, Path]: (詳細結果CSVパス, サマリーCSVパス)
        """
        from .inference.csv_exporter import InferenceCSVExporter

        # CSV出力器を作成（推論ワークスペースを指定）
        csv_exporter = InferenceCSVExporter(
            output_dir=str(self.inference_workspace), logger=self.logger
        )

        # モデル情報の取得
        model_info = self.get_model_info()

        # 精度計算
        accuracy_info = self.calculate_accuracy(predicted_labels, true_labels)

        # 詳細結果のCSV出力
        results_csv = csv_exporter.export_results(
            image_paths=image_paths,
            predicted_labels=predicted_labels,
            true_labels=true_labels,
            confidence_scores=confidence_scores,
            class_names=class_names,
            model_info=model_info,
            filename=results_filename,
        )

        # サマリーのCSV出力
        summary_csv = csv_exporter.export_summary(
            accuracy_info=accuracy_info,
            model_info=model_info,
            filename=summary_filename,
        )

        # モデル情報もJSONで保存
        self.inference_workspace_manager.save_model_info(model_info)

        return results_csv, summary_csv

    def get_inference_workspace_info(self) -> Dict[str, Any]:
        """
        推論ワークスペース情報を取得.

        Returns:
            Dict[str, any]: ワークスペース情報
        """
        return self.inference_workspace_manager.get_workspace_info()
