"""
pochitrain.core.evaluator: 評価器モジュール

モデルの評価とメトリクス計算を担当する評価器モジュール
"""

import logging
from typing import Dict, Any, List, Optional, Union
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class Evaluator:
    """
    評価器クラス

    モデルの評価とメトリクス計算を行う

    Args:
        logger (logging.Logger, optional): ロガー

    Examples:
        >>> evaluator = Evaluator()
        >>> metrics = evaluator.evaluate(model, dataloader, device)
        >>> print(f"精度: {metrics['accuracy']:.2f}%")
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """ロガーの設定"""
        logger = logging.getLogger('pochitrain.evaluator')
        logger.setLevel(logging.INFO)

        # ハンドラーが既に存在する場合は追加しない
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def evaluate(self,
                 model: nn.Module,
                 dataloader: DataLoader,
                 device: torch.device,
                 criterion: Optional[nn.Module] = None) -> Dict[str, Any]:
        """
        モデルの評価を実行

        Args:
            model (nn.Module): 評価するモデル
            dataloader (DataLoader): データローダー
            device (torch.device): デバイス
            criterion (nn.Module, optional): 損失関数

        Returns:
            Dict[str, Any]: 評価結果
        """
        self.logger.info("評価を開始します...")

        model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)

                # 順伝播
                output = model(data)

                # 損失計算
                if criterion:
                    loss = criterion(output, target)
                    total_loss += loss.item()

                # 予測値の取得
                _, predicted = output.max(1)

                # 結果の蓄積
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

                # 進捗ログ
                if batch_idx % 100 == 0:
                    self.logger.debug(f"評価バッチ {batch_idx}/{len(dataloader)}")

        # メトリクス計算
        metrics = self._calculate_metrics(all_predictions, all_targets)

        if criterion:
            metrics['loss'] = total_loss / len(dataloader)

        self.logger.info(f"評価完了 - 精度: {metrics['accuracy']:.2f}%")

        return metrics

    def _calculate_metrics(self, predictions: List[int], targets: List[int]) -> Dict[str, Any]:
        """
        メトリクスの計算

        Args:
            predictions (List[int]): 予測値のリスト
            targets (List[int]): 正解ラベルのリスト

        Returns:
            Dict[str, Any]: 計算されたメトリクス
        """
        predictions = np.array(predictions)
        targets = np.array(targets)

        # 基本メトリクス
        accuracy = accuracy_score(targets, predictions) * 100
        precision = precision_score(
            targets, predictions, average='weighted', zero_division=0) * 100
        recall = recall_score(targets, predictions,
                              average='weighted', zero_division=0) * 100
        f1 = f1_score(targets, predictions, average='weighted', zero_division=0) * 100

        # 混同行列
        cm = confusion_matrix(targets, predictions)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'total_samples': len(targets),
            'correct_samples': int(np.sum(predictions == targets))
        }

        return metrics

    def evaluate_single_image(self,
                              model: nn.Module,
                              image: torch.Tensor,
                              device: torch.device,
                              class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        単一画像の評価

        Args:
            model (nn.Module): 評価するモデル
            image (torch.Tensor): 入力画像
            device (torch.device): デバイス
            class_names (List[str], optional): クラス名のリスト

        Returns:
            Dict[str, Any]: 評価結果
        """
        model.eval()

        # バッチ次元を追加
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(device)

        with torch.no_grad():
            output = model(image)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = probabilities.max(1)

        result = {
            'predicted_class': predicted.item(),
            'confidence': confidence.item() * 100,
            'probabilities': probabilities.cpu().numpy()[0].tolist()
        }

        if class_names:
            result['predicted_class_name'] = class_names[predicted.item()]
            result['class_probabilities'] = {
                class_names[i]: prob for i, prob in enumerate(result['probabilities'])
            }

        return result

    def calculate_top_k_accuracy(self,
                                 predictions: torch.Tensor,
                                 targets: torch.Tensor,
                                 k: int = 5) -> float:
        """
        Top-k精度の計算

        Args:
            predictions (torch.Tensor): 予測値
            targets (torch.Tensor): 正解ラベル
            k (int): Top-k のk

        Returns:
            float: Top-k精度（パーセンテージ）
        """
        _, predicted = predictions.topk(k, 1, True, True)
        predicted = predicted.t()
        correct = predicted.eq(targets.view(1, -1).expand_as(predicted))

        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        accuracy = correct_k.mul_(100.0 / targets.size(0))

        return accuracy.item()

    def analyze_predictions(self,
                            predictions: List[int],
                            targets: List[int],
                            class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        予測結果の詳細分析

        Args:
            predictions (List[int]): 予測値のリスト
            targets (List[int]): 正解ラベルのリスト
            class_names (List[str], optional): クラス名のリスト

        Returns:
            Dict[str, Any]: 分析結果
        """
        predictions = np.array(predictions)
        targets = np.array(targets)

        # 基本統計
        total_samples = len(targets)
        correct_samples = np.sum(predictions == targets)
        accuracy = correct_samples / total_samples * 100

        # クラス別精度
        unique_classes = np.unique(targets)
        class_accuracy = {}
        class_samples = {}

        for cls in unique_classes:
            mask = targets == cls
            class_total = np.sum(mask)
            class_correct = np.sum(predictions[mask] == targets[mask])
            class_acc = class_correct / class_total * 100 if class_total > 0 else 0

            class_key = class_names[cls] if class_names else f"Class_{cls}"
            class_accuracy[class_key] = class_acc
            class_samples[class_key] = class_total

        # 混同行列の詳細分析
        cm = confusion_matrix(targets, predictions)

        # 最も混同されやすいクラスペア
        confusion_pairs = []
        for i in range(len(cm)):
            for j in range(len(cm)):
                if i != j and cm[i, j] > 0:
                    true_class = class_names[i] if class_names else f"Class_{i}"
                    pred_class = class_names[j] if class_names else f"Class_{j}"
                    confusion_pairs.append({
                        'true_class': true_class,
                        'predicted_class': pred_class,
                        'count': int(cm[i, j]),
                        'percentage': cm[i, j] / np.sum(cm[i, :]) * 100
                    })

        # 混同の多い順にソート
        confusion_pairs.sort(key=lambda x: x['count'], reverse=True)

        analysis = {
            'overall_accuracy': accuracy,
            'total_samples': total_samples,
            'correct_samples': correct_samples,
            'class_accuracy': class_accuracy,
            'class_samples': class_samples,
            'confusion_matrix': cm.tolist(),
            'most_confused_pairs': confusion_pairs[:10]  # 上位10個
        }

        return analysis

    def save_evaluation_report(self,
                               metrics: Dict[str, Any],
                               output_path: str) -> None:
        """
        評価レポートの保存

        Args:
            metrics (Dict[str, Any]): 評価メトリクス
            output_path (str): 出力パス
        """
        import json
        from datetime import datetime

        report = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'summary': {
                'accuracy': f"{metrics.get('accuracy', 0):.2f}%",
                'precision': f"{metrics.get('precision', 0):.2f}%",
                'recall': f"{metrics.get('recall', 0):.2f}%",
                'f1_score': f"{metrics.get('f1_score', 0):.2f}%"
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self.logger.info(f"評価レポートを保存しました: {output_path}")

    def print_evaluation_summary(self, metrics: Dict[str, Any]) -> None:
        """
        評価結果の要約を表示

        Args:
            metrics (Dict[str, Any]): 評価メトリクス
        """
        print("\n" + "="*50)
        print("評価結果サマリー")
        print("="*50)
        print(f"総サンプル数: {metrics.get('total_samples', 0):,}")
        print(f"正解サンプル数: {metrics.get('correct_samples', 0):,}")
        print(f"精度 (Accuracy): {metrics.get('accuracy', 0):.2f}%")
        print(f"適合率 (Precision): {metrics.get('precision', 0):.2f}%")
        print(f"再現率 (Recall): {metrics.get('recall', 0):.2f}%")
        print(f"F1スコア: {metrics.get('f1_score', 0):.2f}%")

        if 'loss' in metrics:
            print(f"損失 (Loss): {metrics['loss']:.4f}")

        print("="*50)

        # クラス別精度がある場合
        if 'class_accuracy' in metrics:
            print("\nクラス別精度:")
            print("-"*30)
            for class_name, accuracy in metrics['class_accuracy'].items():
                samples = metrics.get('class_samples', {}).get(class_name, 0)
                print(f"{class_name}: {accuracy:.2f}% ({samples} サンプル)")
            print()

        # 混同の多いクラスペア
        if 'most_confused_pairs' in metrics:
            print("最も混同されやすいクラスペア:")
            print("-"*40)
            for pair in metrics['most_confused_pairs'][:5]:  # 上位5個
                print(f"{pair['true_class']} → {pair['predicted_class']}: "
                      f"{pair['count']} 回 ({pair['percentage']:.1f}%)")
            print()
