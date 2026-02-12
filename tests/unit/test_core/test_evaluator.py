"""Evaluatorクラスのテスト.

PochiTrainer._compute_confusion_matrix_pytorch() と
PochiPredictor.calculate_accuracy() から移行したテストを含む.
"""

import logging
import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pochitrain.training.evaluator import Evaluator


@pytest.fixture
def evaluator():
    """CPU上のEvaluatorインスタンスを返す."""
    logger = logging.getLogger("test_evaluator")
    return Evaluator(device=torch.device("cpu"), logger=logger)


class TestComputeConfusionMatrix:
    """compute_confusion_matrixメソッドのテスト (旧TestConfusionMatrixCalculation)."""

    def test_basic_case(self, evaluator):
        """基本的な混同行列計算."""
        predicted = torch.tensor([0, 1, 2, 3, 0, 1])
        targets = torch.tensor([0, 1, 2, 3, 0, 2])

        cm = evaluator.compute_confusion_matrix(predicted, targets, 4)

        expected = torch.tensor(
            [
                [2, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        assert torch.equal(cm, expected), f"Expected {expected}, but got {cm}"

    def test_perfect_prediction(self, evaluator):
        """完全に正解した場合の混同行列."""
        predicted = torch.tensor([0, 1, 2, 3])
        targets = torch.tensor([0, 1, 2, 3])

        cm = evaluator.compute_confusion_matrix(predicted, targets, 4)

        expected = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        assert torch.equal(cm, expected)

    def test_all_wrong(self, evaluator):
        """全て間違った場合の混同行列."""
        predicted = torch.tensor([1, 2, 3, 0])
        targets = torch.tensor([0, 1, 2, 3])

        cm = evaluator.compute_confusion_matrix(predicted, targets, 4)

        expected = torch.tensor(
            [
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
            ]
        )
        assert torch.equal(cm, expected)

    def test_empty_input(self, evaluator):
        """空の入力に対するテスト."""
        predicted = torch.tensor([])
        targets = torch.tensor([])

        cm = evaluator.compute_confusion_matrix(predicted, targets, 4)

        expected = torch.zeros(4, 4, dtype=torch.int64)
        assert torch.equal(cm, expected)

    def test_single_class(self, evaluator):
        """単一クラスのみの場合."""
        predicted = torch.tensor([2, 2, 2])
        targets = torch.tensor([2, 2, 2])

        cm = evaluator.compute_confusion_matrix(predicted, targets, 4)

        expected = torch.tensor(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 3, 0],
                [0, 0, 0, 0],
            ]
        )
        assert torch.equal(cm, expected)

    def test_device_consistency(self, evaluator):
        """デバイス間での一貫性テスト."""
        predicted = torch.tensor([0, 1, 2, 1])
        targets = torch.tensor([0, 1, 1, 2])

        cm = evaluator.compute_confusion_matrix(predicted, targets, 3)

        assert cm.device.type == "cpu"

        expected = torch.tensor(
            [
                [1, 0, 0],
                [0, 1, 1],
                [0, 1, 0],
            ]
        )
        assert torch.equal(cm, expected)


class TestCalculateAccuracy:
    """calculate_accuracyメソッドのテスト (旧TestPochiPredictorCalculateAccuracy)."""

    def test_perfect_accuracy(self, evaluator):
        """全問正解で100%."""
        result = evaluator.calculate_accuracy([0, 1, 2], [0, 1, 2])
        assert result["accuracy_percentage"] == 100.0
        assert result["correct_predictions"] == 3
        assert result["total_samples"] == 3

    def test_zero_accuracy(self, evaluator):
        """全問不正解で0%."""
        result = evaluator.calculate_accuracy([1, 2, 0], [0, 1, 2])
        assert result["accuracy_percentage"] == 0.0
        assert result["correct_predictions"] == 0

    def test_partial_accuracy(self, evaluator):
        """部分正解."""
        result = evaluator.calculate_accuracy([0, 1, 0, 1], [0, 0, 0, 1])
        assert result["accuracy_percentage"] == 75.0
        assert result["correct_predictions"] == 3

    def test_empty_lists(self, evaluator):
        """空のリストで0%."""
        result = evaluator.calculate_accuracy([], [])
        assert result["accuracy_percentage"] == 0.0
        assert result["total_samples"] == 0


class TestValidate:
    """validateメソッドのテスト."""

    def test_validate_basic(self, evaluator):
        """バリデーションループの正常動作."""
        # 簡易モデル: 3クラス分類
        model = nn.Linear(4, 3)
        model.eval()
        criterion = nn.CrossEntropyLoss()

        # テストデータ: 8サンプル
        data = torch.randn(8, 4)
        targets = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
        dataset = TensorDataset(data, targets)
        loader = DataLoader(dataset, batch_size=4)

        result = evaluator.validate(model, loader, criterion)

        assert "val_loss" in result
        assert "val_accuracy" in result
        assert result["val_loss"] >= 0.0
        assert 0.0 <= result["val_accuracy"] <= 100.0

    def test_validate_with_confusion_matrix(self, evaluator):
        """混同行列計算付きのバリデーション."""
        model = nn.Linear(4, 3)
        model.eval()
        criterion = nn.CrossEntropyLoss()

        data = torch.randn(6, 4)
        targets = torch.tensor([0, 1, 2, 0, 1, 2])
        dataset = TensorDataset(data, targets)
        loader = DataLoader(dataset, batch_size=3)

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = evaluator.validate(
                model,
                loader,
                criterion,
                num_classes_for_cm=3,
                epoch=1,
                workspace_path=Path(tmp_dir),
            )

            assert "val_loss" in result
            assert "val_accuracy" in result
            # 混同行列ログファイルが作成されていることを確認
            log_file = Path(tmp_dir) / "confusion_matrix.log"
            assert log_file.exists()


class TestSampleWeightedLoss:
    """サンプル重み付け損失平均のテスト."""

    def test_validate_sample_weighted_loss(self, evaluator):
        """不均一バッチサイズでサンプル重み付け平均が正しく計算される."""
        # 固定重みのモデルで再現性を確保
        torch.manual_seed(42)
        model = nn.Linear(4, 3)
        model.eval()
        criterion = nn.CrossEntropyLoss()

        # 7サンプル, batch_size=4 -> バッチ1: 4サンプル, バッチ2: 3サンプル
        data = torch.randn(7, 4)
        targets = torch.tensor([0, 1, 2, 0, 1, 2, 0])
        dataset = TensorDataset(data, targets)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)

        result = evaluator.validate(model, loader, criterion)

        # 手動で期待値を計算
        with torch.no_grad():
            out1 = model(data[:4])
            loss1 = criterion(out1, targets[:4]).item()
            out2 = model(data[4:])
            loss2 = criterion(out2, targets[4:]).item()

        expected_loss = (loss1 * 4 + loss2 * 3) / 7
        assert result["val_loss"] == pytest.approx(expected_loss, abs=1e-6)


class TestLogConfusionMatrix:
    """log_confusion_matrixメソッドのテスト."""

    def test_log_creates_file(self, evaluator):
        """混同行列ログファイルが作成される."""
        cm = torch.tensor([[2, 1], [0, 3]], dtype=torch.int64)

        with tempfile.TemporaryDirectory() as tmp_dir:
            evaluator.log_confusion_matrix(cm, epoch=1, workspace_path=Path(tmp_dir))
            log_file = Path(tmp_dir) / "confusion_matrix.log"
            assert log_file.exists()
            content = log_file.read_text(encoding="utf-8")
            assert "epoch1" in content

    def test_log_skips_when_no_workspace(self, evaluator):
        """workspace_path=None のときスキップする."""
        cm = torch.tensor([[1, 0], [0, 1]], dtype=torch.int64)
        # 例外が発生しないことを確認
        evaluator.log_confusion_matrix(cm, epoch=1, workspace_path=None)
