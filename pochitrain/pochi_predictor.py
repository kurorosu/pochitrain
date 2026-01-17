"""
pochitrain.pochi_predictor: 推論機能のメインモジュール.

学習済みモデルを読み込んで推論を実行する機能を提供します。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
        # 推論専用ワークスペースマネージャーを作成（遅延作成）
        self.inference_workspace_manager = InferenceWorkspaceManager(work_dir)
        self.inference_workspace: Optional[Path] = None  # 遅延作成のためNoneで初期化

        # 親クラスを初期化（推論モードではワークスペース作成をスキップ）
        super().__init__(
            model_name=model_name,
            num_classes=num_classes,
            device=device,
            pretrained=False,
            work_dir=work_dir,
            create_workspace=False,  # 推論時はワークスペース作成をスキップ
        )

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

    def _ensure_inference_workspace(self) -> Path:
        """
        必要時にワークスペースを作成.

        遅延作成パターンにより、実際にワークスペースが必要になったタイミングで
        InferenceWorkspaceManagerを使用してワークスペースを作成します。

        Returns:
            Path: 作成されたワークスペースのパス
        """
        if self.inference_workspace is None:
            self.inference_workspace = (
                self.inference_workspace_manager.create_workspace()
            )
            self.logger.info(f"推論ワークスペースを作成: {self.inference_workspace}")
        return self.inference_workspace

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
        cm_config: Optional[Dict[str, Any]] = None,
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

        # 推論ワークスペースを確保（遅延作成）
        inference_workspace = self._ensure_inference_workspace()

        # CSV出力器を作成（推論ワークスペースを指定）
        csv_exporter = InferenceCSVExporter(
            output_dir=str(inference_workspace), logger=self.logger
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

        # 混同行列画像を自動生成
        try:
            confusion_matrix_path = self.save_confusion_matrix_image(
                predicted_labels=predicted_labels,
                true_labels=true_labels,
                class_names=class_names,
                cm_config=cm_config,
            )
            self.logger.info(f"混同行列画像も生成されました: {confusion_matrix_path}")
        except Exception as e:
            self.logger.warning(f"混同行列画像生成に失敗しました: {e}")

        return results_csv, summary_csv

    def get_inference_workspace_info(self) -> Dict[str, Any]:
        """
        推論ワークスペース情報を取得.

        Returns:
            Dict[str, any]: ワークスペース情報
        """
        # ワークスペースが作成されている場合のみ情報を取得
        if self.inference_workspace is not None:
            return self.inference_workspace_manager.get_workspace_info()
        else:
            return {
                "workspace_path": None,
                "workspace_name": None,
                "exists": False,
            }

    def save_confusion_matrix_image(
        self,
        predicted_labels: List[int],
        true_labels: List[int],
        class_names: List[str],
        filename: str = "confusion_matrix.png",
        cm_config: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        混同行列の画像を保存.

        Args:
            predicted_labels (List[int]): 予測ラベルのリスト
            true_labels (List[int]): 正解ラベルのリスト
            class_names (List[str]): クラス名のリスト
            filename (str): 保存ファイル名
            cm_config (Optional[Dict[str, Any]]): 混同行列可視化設定

        Returns:
            Path: 保存されたファイルのパス
        """
        import matplotlib.pyplot as plt
        import matplotlib_fontja  # noqa: F401
        import numpy as np

        # デフォルト設定
        default_config = {
            "title": "Confusion Matrix",
            "xlabel": "Predicted Label",
            "ylabel": "True Label",
            "fontsize": 14,
            "title_fontsize": 16,
            "label_fontsize": 12,
            "figsize": (8, 6),
            "cmap": "Blues",
        }

        # 設定をマージ（cm_configが指定されていれば優先）
        config = default_config.copy()
        if cm_config:
            config.update(cm_config)

        # リストをPyTorchテンソルに変換
        predicted = torch.tensor(predicted_labels, dtype=torch.long, device=self.device)
        targets = torch.tensor(true_labels, dtype=torch.long, device=self.device)

        # 継承した混同行列計算メソッドを使用
        cm_tensor = self._compute_confusion_matrix_pytorch(
            predicted, targets, len(class_names)
        )
        cm = cm_tensor.cpu().numpy()

        # プロット作成（設定から図のサイズを取得）
        fig, ax = plt.subplots(figsize=config["figsize"])

        # ヒートマップを描画（設定からカラーマップを取得）
        cmap_value: str = str(config["cmap"])
        ax.imshow(cm, interpolation="nearest", cmap=cmap_value)

        # ラベル設定
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)

        # ラベルとタイトル（設定から取得）
        xlabel: str = str(config["xlabel"])
        ylabel: str = str(config["ylabel"])
        title: str = str(config["title"])
        ax.set_xlabel(xlabel, fontsize=config["label_fontsize"])
        ax.set_ylabel(ylabel, fontsize=config["label_fontsize"])
        ax.set_title(title, fontsize=config["title_fontsize"])

        # 各セルに数値を表示（設定からフォントサイズを取得）
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=config["fontsize"],
                )

        # レイアウト調整
        plt.tight_layout()

        # 推論ワークスペースを確保（遅延作成）
        inference_workspace = self._ensure_inference_workspace()

        # ファイル保存
        output_path = inference_workspace / filename
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        self.logger.info(f"混同行列画像保存: {output_path}")
        return output_path
