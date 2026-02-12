"""
訓練メトリクスのCSV出力とグラフ可視化機能.

訓練時のメトリクス（学習率、損失、精度）を記録し、
CSV形式で保存してグラフを自動生成します。
"""

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

# 画像ファイル出力のみを行うため、GUIバックエンド依存を避ける.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pochitrain.exporters import BaseCSVExporter


class TrainingMetricsExporter(BaseCSVExporter):
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
        layer_wise_lr_graph_config: Optional[Dict[str, Any]] = None,
    ):
        """TrainingMetricsExporterを初期化."""
        super().__init__(output_dir=output_dir, logger=logger)
        self.enable_visualization = enable_visualization

        # 層別学習率グラフの設定
        if layer_wise_lr_graph_config is None:
            layer_wise_lr_graph_config = {}
        self.layer_wise_lr_graph_config = layer_wise_lr_graph_config
        self.use_log_scale = layer_wise_lr_graph_config.get("use_log_scale", True)

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
        layer_wise_lr_enabled: bool = False,
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
            layer_wise_lr_enabled (bool): 層別学習率が有効かどうか
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

        # 層別学習率の状態を記録
        metrics["layer_wise_lr_enabled"] = layer_wise_lr_enabled

        # 層別学習率のカラムを動的に追加
        if layer_wise_lr_enabled:
            for key in kwargs:
                if key.startswith("lr_") and key not in self.extended_headers:
                    self.add_extended_headers([key])

        self.metrics_history.append(metrics)

        # ログ出力（層別学習率対応）
        if layer_wise_lr_enabled:
            lr_display = f"{learning_rate:.6f} (層別設定)"
        else:
            lr_display = f"{learning_rate:.6f}"

        self.logger.debug(
            f"エポック {epoch} のメトリクスを記録: "
            f"LR={lr_display}, Loss={train_loss:.4f}, Acc={train_accuracy:.2f}%"
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

        filename = self._generate_filename("training_metrics", filename)
        output_path = self._build_output_path(filename)

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
        val_losses: List[float] = []
        val_accuracies: List[float] = []
        if has_val_data:
            val_losses = [
                float(m["val_loss"])
                for m in self.metrics_history
                if m["val_loss"] != ""
            ]
            val_accuracies = [
                float(m["val_accuracy"])
                for m in self.metrics_history
                if m["val_accuracy"] != ""
            ]

        # ベースファイル名の生成
        if base_filename is None:
            timestamp = self._generate_filename("training_metrics")
            # .csv を除いてベース名として使う
            base_filename = timestamp.removesuffix(".csv")

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

        # 2. 精度推移グラフ（層別学習率対応）
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

        # 層別学習率が有効かどうかを判定（メトリクス履歴から取得）
        layer_wise_lr_active = any(
            m.get("layer_wise_lr_enabled", False) for m in self.metrics_history
        )

        if not layer_wise_lr_active:
            # 通常の場合：学習率を第2軸にプロット
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
            ax1.legend(
                lines1 + lines2, labels1 + labels2, loc="center right", fontsize=11
            )

            title = "Accuracy and Learning Rate"
        else:
            # 層別学習率有効時：学習率を表示しない
            ax1.legend(loc="center right", fontsize=11)
            title = "Accuracy (Layer-wise LR Active)"

        ax1.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        acc_path = self.output_dir / f"{base_filename}_accuracy.png"
        plt.savefig(acc_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(acc_path)
        self.logger.info(f"精度・学習率グラフを生成: {acc_path}")

        # 3. 層別学習率グラフ（層別学習率有効時のみ）
        if layer_wise_lr_active:
            lr_graph_path = self._generate_layer_wise_lr_graph(base_filename)
            if lr_graph_path:
                output_paths.append(lr_graph_path)

        return output_paths

    def _generate_layer_wise_lr_graph(self, base_filename: str) -> Optional[Path]:
        """
        層別学習率専用のグラフを生成.

        Args:
            base_filename (str): ベースファイル名

        Returns:
            Optional[Path]: 生成されたグラフファイルのパス
        """
        if not self.metrics_history:
            return None

        # 層別学習率のカラムを抽出
        lr_columns = []
        for key in self.metrics_history[0].keys():
            if key.startswith("lr_"):
                lr_columns.append(key)

        if not lr_columns:
            return None

        epochs = [m["epoch"] for m in self.metrics_history]

        # グラフ作成
        fig, ax = plt.subplots(figsize=(12, 8))

        # 各層の学習率をプロット
        colors = plt.get_cmap("tab10")(range(len(lr_columns)))
        for i, lr_col in enumerate(lr_columns):
            layer_name = lr_col.replace("lr_", "")
            learning_rates = [m.get(lr_col, 0) for m in self.metrics_history]
            ax.plot(
                epochs,
                learning_rates,
                color=colors[i],
                linewidth=2,
                marker="o",
                markersize=4,
                label=f"{layer_name}",
            )

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Learning Rate", fontsize=12)
        ax.set_title("Layer-wise Learning Rates", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=10)
        # 対数スケールの使用を設定から制御
        if self.use_log_scale:
            ax.set_yscale("log")
        plt.tight_layout()

        lr_path = self.output_dir / f"{base_filename}_layer_wise_lr.png"
        plt.savefig(lr_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        self.logger.info(f"層別学習率グラフを生成: {lr_path}")
        return lr_path

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
