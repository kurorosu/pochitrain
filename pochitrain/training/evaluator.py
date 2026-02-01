"""検証・混同行列計算・精度計算を担当するモジュール."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader


class Evaluator:
    """検証・混同行列計算・精度計算を担当するクラス.

    Args:
        device: 計算デバイス
        logger: ロガーインスタンス
    """

    def __init__(self, device: torch.device, logger: logging.Logger) -> None:
        """Evaluatorを初期化."""
        self.device = device
        self.logger = logger

    def validate(
        self,
        model: nn.Module,
        val_loader: DataLoader[Any],
        criterion: nn.Module,
        num_classes_for_cm: Optional[int] = None,
        epoch: int = 0,
        workspace_path: Optional[Path] = None,
    ) -> Dict[str, float]:
        """検証.

        Args:
            model: 検証対象のモデル
            val_loader: 検証データローダー
            criterion: 損失関数
            num_classes_for_cm: 混同行列計算用のクラス数 (Noneなら混同行列をスキップ)
            epoch: 現在のエポック番号 (混同行列ログ用)
            workspace_path: ワークスペースパス (混同行列ログ出力先)

        Returns:
            検証メトリクス {"val_loss": float, "val_accuracy": float}
        """
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        # 混同行列計算のためのリスト
        all_predicted: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = model(data)
                loss = criterion(output, target)

                batch_size = target.size(0)
                total_loss += loss.item() * batch_size
                _, predicted = output.max(1)
                total += batch_size
                correct += predicted.eq(target).sum().item()

                # 混同行列用にデータを保存
                all_predicted.append(predicted)
                all_targets.append(target)

        # 例外回避のための防御的ガード. 本来はバリデーションで止めるのが望ましい
        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = 100.0 * correct / total if total > 0 else 0.0

        # 混同行列の計算と出力
        if num_classes_for_cm is not None and all_predicted and all_targets:
            all_predicted_tensor = torch.cat(all_predicted, dim=0)
            all_targets_tensor = torch.cat(all_targets, dim=0)

            cm = self.compute_confusion_matrix(
                all_predicted_tensor, all_targets_tensor, num_classes_for_cm
            )

            self.log_confusion_matrix(cm, epoch, workspace_path)

        return {"val_loss": avg_loss, "val_accuracy": accuracy}

    def compute_confusion_matrix(
        self, predicted: torch.Tensor, targets: torch.Tensor, num_classes: int
    ) -> torch.Tensor:
        """純粋なPyTorchテンソル操作による混同行列計算.

        sklearn.metrics.confusion_matrixやtorchmetricsを使用せず,
        基本的なPyTorchテンソル操作のみで混同行列を計算します.
        これにより, Ctrl+C割り込み時のFortranランタイムエラーを回避できます.

        Args:
            predicted: 予測ラベル (各要素は0からnum_classes-1の整数)
            targets: 正解ラベル (各要素は0からnum_classes-1の整数)
            num_classes: クラス数

        Returns:
            混同行列 ([num_classes, num_classes]の2次元テンソル)
            行が正解ラベル, 列が予測ラベルに対応.
        """
        predicted = predicted.to(self.device)
        targets = targets.to(self.device)

        confusion_matrix = torch.zeros(
            num_classes, num_classes, dtype=torch.int64, device=self.device
        )

        for t, p in zip(targets.view(-1), predicted.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

        return confusion_matrix

    def log_confusion_matrix(
        self,
        confusion_matrix: torch.Tensor,
        epoch: int,
        workspace_path: Optional[Path] = None,
    ) -> None:
        """混同行列をログファイルに出力.

        Args:
            confusion_matrix: PyTorchテンソル形式の混同行列
            epoch: エポック番号
            workspace_path: ワークスペースパス (Noneならスキップ)
        """
        if workspace_path is None:
            return

        log_file = Path(workspace_path) / "confusion_matrix.log"

        cm_numpy = confusion_matrix.cpu().numpy().astype(int)

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"epoch{epoch}\n")
            for row in cm_numpy:
                f.write(f" {row.tolist()}\n")

    def calculate_accuracy(
        self, predicted_labels: List[int], true_labels: List[int]
    ) -> Dict[str, float]:
        """推論結果の精度を計算.

        Args:
            predicted_labels: 推論ラベル
            true_labels: 正解ラベル

        Returns:
            精度情報
        """
        total = len(predicted_labels)
        correct = sum(p == t for p, t in zip(predicted_labels, true_labels))
        accuracy = (correct / total) * 100 if total > 0 else 0.0

        accuracy_info: Dict[str, float] = {
            "total_samples": total,
            "correct_predictions": correct,
            "accuracy_percentage": accuracy,
        }

        self.logger.info(f"推論精度: {correct}/{total} ({accuracy:.2f}%)")

        return accuracy_info
