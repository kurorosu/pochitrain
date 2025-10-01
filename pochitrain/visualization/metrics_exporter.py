"""
訓練メトリクスのCSV出力とグラフ可視化機能.

訓練時のメトリクス（学習率、損失、精度）を記録し、
CSV形式で保存してグラフを自動生成します。
"""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt


class TrainingMetricsExporter:
    """
    訓練メトリクスをCSVファイルに出力し、グラフを生成するクラス.

    Args:
        output_dir (Path): 出力ディレクトリ（visualization/）
        enable_visualization (bool): グラフ生成を有効にするか
        logger (logging.Logger, optional): ロガーインスタンス
    """

    def __init__(
        self,
        output_dir: Path,
        enable_visualization: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        """TrainingMetricsExporterを初期化."""
        self.output_dir = Path(output_dir)
        self.enable_visualization = enable_visualization

        # ロガーの設定
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

        # メトリクス履歴の初期化
        self.metrics_history: List[Dict[str, Any]] = []

        # CSVヘッダー定義（拡張可能な設計）
        self.base_headers = [
            "epoch",
            "learning_rate",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
        ]
        self.extended_headers: List[str] = []  # Issue 9用の拡張ヘッダー

    def add_extended_headers(self, headers: List[str]) -> None:
        """
        拡張ヘッダーを追加（Issue 9でのパラメータ追跡用）.

        Args:
            headers (List[str]): 追加するヘッダー名のリスト
        """
        self.extended_headers.extend(headers)
        self.logger.debug(f"拡張ヘッダーを追加: {headers}")

    def record_epoch(
        self,
        epoch: int,
        learning_rate: float,
        train_loss: float,
        train_accuracy: float,
        val_loss: Optional[float] = None,
        val_accuracy: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        エポック毎のメトリクスを記録.

        Args:
            epoch (int): エポック番号
            learning_rate (float): 学習率
            train_loss (float): 訓練損失
            train_accuracy (float): 訓練精度
            val_loss (float, optional): 検証損失
            val_accuracy (float, optional): 検証精度
            **kwargs: 拡張メトリクス（Issue 9用のパラメータ値など）
        """
        metrics = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss if val_loss is not None else "",
            "val_accuracy": val_accuracy if val_accuracy is not None else "",
        }

        # 拡張メトリクスを追加
        for key, value in kwargs.items():
            metrics[key] = value

        self.metrics_history.append(metrics)
        self.logger.debug(
            f"エポック {epoch} のメトリクスを記録: "
            f"LR={learning_rate:.6f}, Loss={train_loss:.4f}, Acc={train_accuracy:.2f}%"
        )

    def export_to_csv(self, filename: Optional[str] = None) -> Optional[Path]:
        """
        メトリクスをCSVファイルに出力.

        Args:
            filename (str, optional): 出力ファイル名

        Returns:
            Optional[Path]: 出力されたCSVファイルのパス（記録がない場合はNone）
        """
        if not self.metrics_history:
            self.logger.warning("記録されたメトリクスがありません")
            return None

        # ファイル名の生成
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_metrics_{timestamp}.csv"

        if not filename.endswith(".csv"):
            filename += ".csv"

        output_path = self.output_dir / filename

        # 全ヘッダーの結合
        all_headers = self.base_headers + self.extended_headers

        # CSVファイルの書き込み
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=all_headers)
            writer.writeheader()

            for metrics in self.metrics_history:
                # 存在しないキーには空文字を設定
                row = {header: metrics.get(header, "") for header in all_headers}
                writer.writerow(row)

        self.logger.info(f"訓練メトリクスをCSVに出力: {output_path}")
        return output_path

    def generate_graphs(
        self, base_filename: Optional[str] = None
    ) -> Optional[List[Path]]:
        """
        学習率推移、損失推移、精度推移のグラフを個別ファイルとして生成.

        Args:
            base_filename (str, optional): 出力ファイル名のベース（拡張子なし）

        Returns:
            Optional[List[Path]]: 出力されたグラフファイルのパスリスト（生成しない場合はNone）
        """
        if not self.enable_visualization:
            self.logger.debug("グラフ生成が無効化されています")
            return None

        if not self.metrics_history:
            self.logger.warning("記録されたメトリクスがありません")
            return None

        # データの抽出
        epochs = [m["epoch"] for m in self.metrics_history]
        learning_rates = [m["learning_rate"] for m in self.metrics_history]
        train_losses = [m["train_loss"] for m in self.metrics_history]
        train_accuracies = [m["train_accuracy"] for m in self.metrics_history]

        # 検証データがある場合のみ抽出
        has_val_data = any(m["val_loss"] != "" for m in self.metrics_history)
        if has_val_data:
            val_losses = [
                m["val_loss"] if m["val_loss"] != "" else None
                for m in self.metrics_history
            ]
            val_accuracies = [
                m["val_accuracy"] if m["val_accuracy"] != "" else None
                for m in self.metrics_history
            ]

        # ベースファイル名の生成
        if base_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"training_metrics_{timestamp}"

        output_paths = []

        # 1. 損失推移グラフ
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            epochs,
            train_losses,
            "r-",
            linewidth=2,
            marker="o",
            markersize=4,
            label="Train Loss",
        )
        if has_val_data:
            ax.plot(
                epochs,
                val_losses,
                "b-",
                linewidth=2,
                marker="s",
                markersize=4,
                label="Validation Loss",
            )
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Loss Curves", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=11)
        plt.tight_layout()

        loss_path = self.output_dir / f"{base_filename}_loss.png"
        plt.savefig(loss_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(loss_path)
        self.logger.info(f"損失グラフを生成: {loss_path}")

        # 2. 精度推移グラフ（学習率を第2軸に統合）
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # 精度を第1軸にプロット
        color_train = "tab:red"
        color_val = "tab:blue"
        ax1.plot(
            epochs,
            train_accuracies,
            color=color_train,
            linewidth=2,
            marker="o",
            markersize=4,
            label="Train Accuracy",
        )
        if has_val_data:
            ax1.plot(
                epochs,
                val_accuracies,
                color=color_val,
                linewidth=2,
                marker="s",
                markersize=4,
                label="Validation Accuracy",
            )
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Accuracy (%)", fontsize=12)
        ax1.tick_params(axis="y")
        ax1.grid(True, alpha=0.3)

        # 学習率を第2軸にプロット
        ax2 = ax1.twinx()
        color_lr = "tab:green"
        ax2.plot(
            epochs,
            learning_rates,
            color=color_lr,
            linewidth=2,
            linestyle="--",
            marker="^",
            markersize=4,
            label="Learning Rate",
        )
        ax2.set_ylabel("Learning Rate", fontsize=12, color=color_lr)
        ax2.tick_params(axis="y", labelcolor=color_lr)

        # 凡例を統合
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=11)

        ax1.set_title("Accuracy and Learning Rate", fontsize=14, fontweight="bold")
        plt.tight_layout()

        acc_path = self.output_dir / f"{base_filename}_accuracy.png"
        plt.savefig(acc_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(acc_path)
        self.logger.info(f"精度・学習率グラフを生成: {acc_path}")

        return output_paths

    def export_all(
        self,
        csv_filename: Optional[str] = None,
        graph_base_filename: Optional[str] = None,
    ) -> tuple[Optional[Path], Optional[List[Path]]]:
        """
        CSVとグラフの両方をエクスポート.

        Args:
            csv_filename (str, optional): CSV出力ファイル名
            graph_base_filename (str, optional): グラフ出力ファイル名のベース

        Returns:
            tuple[Optional[Path], Optional[List[Path]]]: (CSVファイルパス, グラフファイルパスリスト)
        """
        csv_path = self.export_to_csv(csv_filename)
        graph_paths = self.generate_graphs(graph_base_filename)

        return csv_path, graph_paths

    def get_best_epoch(self, metric: str = "val_accuracy") -> Optional[Dict[str, Any]]:
        """
        指定されたメトリクスで最良のエポックを取得.

        Args:
            metric (str): 評価メトリクス名

        Returns:
            Dict[str, Any]: 最良エポックのメトリクス
        """
        if not self.metrics_history:
            return None

        valid_metrics = [m for m in self.metrics_history if m.get(metric, "") != ""]

        if not valid_metrics:
            return None

        # 精度系は最大、損失系は最小を取得
        if "accuracy" in metric.lower():
            best = max(valid_metrics, key=lambda x: x[metric])
        else:
            best = min(valid_metrics, key=lambda x: x[metric])

        return best

    def get_summary(self) -> Dict[str, Any]:
        """
        訓練メトリクスのサマリーを取得.

        Returns:
            Dict[str, Any]: サマリー情報
        """
        if not self.metrics_history:
            return {}

        summary = {
            "total_epochs": len(self.metrics_history),
            "final_train_loss": self.metrics_history[-1]["train_loss"],
            "final_train_accuracy": self.metrics_history[-1]["train_accuracy"],
        }

        # 検証データがある場合
        if self.metrics_history[-1]["val_loss"] != "":
            summary["final_val_loss"] = self.metrics_history[-1]["val_loss"]
            summary["final_val_accuracy"] = self.metrics_history[-1]["val_accuracy"]

            best_val = self.get_best_epoch("val_accuracy")
            if best_val:
                summary["best_val_accuracy"] = best_val["val_accuracy"]
                summary["best_val_accuracy_epoch"] = best_val["epoch"]

        return summary
