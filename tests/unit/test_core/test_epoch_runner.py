"""epoch_runner.py のテスト.

実際の PyTorch モデル, オプティマイザ, DataLoader を使用した古典派テスト.
"""

import logging
from typing import Any

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pochitrain.training.epoch_runner import EpochRunner

DEVICE = torch.device("cpu")


def _make_runner() -> EpochRunner:
    """テスト用 EpochRunner を生成する."""
    logger = logging.getLogger("test.epoch_runner")
    return EpochRunner(device=DEVICE, logger=logger)


def _make_linear_model(in_features: int = 4, num_classes: int = 2) -> nn.Module:
    """テスト用の単純な線形モデルを生成する."""
    return nn.Linear(in_features, num_classes)


def _make_loader(
    num_samples: int = 8,
    in_features: int = 4,
    num_classes: int = 2,
    batch_size: int = 4,
) -> DataLoader[Any]:
    """テスト用の DataLoader を生成する."""
    x = torch.randn(num_samples, in_features)
    y = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size)


class TestEpochRunnerSingleBatch:
    """単一バッチの訓練テスト."""

    def test_loss_is_computed(self):
        """単一バッチの訓練時に損失が正しく計算される."""
        model = _make_linear_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        loader = _make_loader(num_samples=4, batch_size=4)  # 1バッチ
        runner = _make_runner()

        result = runner.run(model, optimizer, criterion, loader, epoch=1)

        assert "loss" in result
        assert "accuracy" in result
        assert result["loss"] > 0.0
        assert 0.0 <= result["accuracy"] <= 100.0


class TestEpochRunnerMultipleBatches:
    """複数バッチの平均損失計算テスト."""

    def test_average_loss_over_batches(self):
        """複数バッチにわたる平均損失の計算が正しい."""
        model = _make_linear_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        # 12サンプル / batch_size=4 = 3バッチ
        loader = _make_loader(num_samples=12, batch_size=4)
        runner = _make_runner()

        result = runner.run(model, optimizer, criterion, loader, epoch=1)

        assert result["loss"] > 0.0
        assert 0.0 <= result["accuracy"] <= 100.0

    def test_loss_is_sample_weighted_average(self):
        """平均損失がサンプル数で重み付けされている."""
        torch.manual_seed(42)
        model = _make_linear_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)  # lr=0 で重み固定
        criterion = nn.CrossEntropyLoss()
        # 不均等バッチ: 5サンプル / batch_size=3 → [3, 2]
        loader = _make_loader(num_samples=5, batch_size=3)
        runner = _make_runner()

        result = runner.run(model, optimizer, criterion, loader, epoch=1)

        # 手動で期待値を計算
        model.eval()
        total_loss = 0.0
        total = 0
        with torch.no_grad():
            for data, target in loader:
                output = model(data)
                loss = criterion(output, target)
                total_loss += loss.item() * target.size(0)
                total += target.size(0)
        expected_avg = total_loss / total

        # lr=0 なので run() の結果と手動計算が一致するはず
        assert result["loss"] == pytest.approx(expected_avg, rel=1e-5)


class TestEpochRunnerEmptyLoader:
    """空の DataLoader に対するテスト."""

    def test_empty_loader_returns_zero(self):
        """空の DataLoader に対して loss=0.0, accuracy=0.0 を返す."""
        model = _make_linear_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        empty_dataset = TensorDataset(
            torch.empty(0, 4), torch.empty(0, dtype=torch.long)
        )
        empty_loader: DataLoader[Any] = DataLoader(empty_dataset, batch_size=1)
        runner = _make_runner()

        result = runner.run(model, optimizer, criterion, empty_loader, epoch=1)

        assert result["loss"] == 0.0
        assert result["accuracy"] == 0.0


class TestEpochRunnerClassWeights:
    """クラス重み付き損失のテスト."""

    def test_class_weighted_loss(self):
        """sample_weighted_loss が有効な場合にクラス重み付き損失が使用される."""
        torch.manual_seed(0)
        model = _make_linear_model()
        optimizer_equal = torch.optim.SGD(model.parameters(), lr=0.0)  # lr=0 で固定

        # 重みなし
        criterion_equal = nn.CrossEntropyLoss()
        loader = _make_loader(num_samples=8, batch_size=4)
        runner = _make_runner()
        result_equal = runner.run(
            model, optimizer_equal, criterion_equal, loader, epoch=1
        )

        # クラス0 に大きな重みを付ける
        criterion_weighted = nn.CrossEntropyLoss(weight=torch.tensor([10.0, 1.0]))
        torch.manual_seed(0)
        model2 = _make_linear_model()
        optimizer_weighted = torch.optim.SGD(model2.parameters(), lr=0.0)
        # 同じデータを使う
        torch.manual_seed(0)
        loader2 = _make_loader(num_samples=8, batch_size=4)
        runner2 = _make_runner()
        result_weighted = runner2.run(
            model2, optimizer_weighted, criterion_weighted, loader2, epoch=1
        )

        # 重み付き損失は重みなしと異なるはず
        assert result_weighted["loss"] != result_equal["loss"]


class TestEpochRunnerGradientUpdate:
    """勾配計算・更新のテスト."""

    def test_parameters_are_updated(self):
        """勾配が正しく計算され, パラメータが更新される."""
        torch.manual_seed(0)
        model = _make_linear_model()
        params_before = [p.clone() for p in model.parameters()]

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss()
        loader = _make_loader(num_samples=8, batch_size=4)
        runner = _make_runner()

        runner.run(model, optimizer, criterion, loader, epoch=1)

        params_after = list(model.parameters())
        any_changed = any(
            not torch.equal(before, after)
            for before, after in zip(params_before, params_after)
        )
        assert any_changed, "訓練後にパラメータが更新されていない"

    def test_parameters_not_updated_with_zero_lr(self):
        """lr=0 の場合パラメータが更新されない."""
        torch.manual_seed(0)
        model = _make_linear_model()
        params_before = [p.clone() for p in model.parameters()]

        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        criterion = nn.CrossEntropyLoss()
        loader = _make_loader(num_samples=8, batch_size=4)
        runner = _make_runner()

        runner.run(model, optimizer, criterion, loader, epoch=1)

        params_after = list(model.parameters())
        all_same = all(
            torch.equal(before, after)
            for before, after in zip(params_before, params_after)
        )
        assert all_same, "lr=0 なのにパラメータが更新されている"
