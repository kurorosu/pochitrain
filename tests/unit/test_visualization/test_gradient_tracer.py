"""GradientTracerクラスのテスト."""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from pochitrain.visualization import GradientTracer


class SimpleModel(nn.Module):
    """テスト用の簡単なモデル."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


@pytest.fixture
def model():
    """テスト用モデルのフィクスチャ."""
    return SimpleModel()


@pytest.fixture
def gradient_tracer():
    """GradientTracerのフィクスチャ."""
    return GradientTracer()


def test_record_gradients(model, gradient_tracer):
    """勾配記録のテスト."""
    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))
    output = model(x)
    loss = nn.functional.cross_entropy(output, y)
    loss.backward()

    gradient_tracer.record_gradients(model, epoch=1)

    assert len(gradient_tracer.epochs) == 1
    assert gradient_tracer.epochs[0] == 1
    assert len(gradient_tracer.layer_names) == 4
    assert len(gradient_tracer.gradient_history) == 4


def test_multiple_epochs(model, gradient_tracer):
    """複数エポックの記録テスト."""
    for epoch in range(1, 4):
        x = torch.randn(4, 10)
        y = torch.randint(0, 2, (4,))
        output = model(x)
        loss = nn.functional.cross_entropy(output, y)
        loss.backward()

        gradient_tracer.record_gradients(model, epoch=epoch)

    assert len(gradient_tracer.epochs) == 3
    assert gradient_tracer.epochs == [1, 2, 3]
    for layer_name in gradient_tracer.layer_names:
        assert len(gradient_tracer.gradient_history[layer_name]) == 3


def test_save_csv(model, gradient_tracer):
    """CSV保存のテスト."""
    for epoch in range(1, 4):
        x = torch.randn(4, 10)
        y = torch.randint(0, 2, (4,))
        output = model(x)
        loss = nn.functional.cross_entropy(output, y)
        loss.backward()
        gradient_tracer.record_gradients(model, epoch=epoch)

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "gradient_trace.csv"
        gradient_tracer.save_csv(csv_path)

        assert csv_path.exists()

        with open(csv_path, "r") as f:
            lines = f.readlines()
            assert len(lines) == 4  # ヘッダー + 3エポック
            assert "epoch" in lines[0]


def test_get_summary(model, gradient_tracer):
    """サマリー取得のテスト."""
    for epoch in range(1, 4):
        x = torch.randn(4, 10)
        y = torch.randint(0, 2, (4,))
        output = model(x)
        loss = nn.functional.cross_entropy(output, y)
        loss.backward()
        gradient_tracer.record_gradients(model, epoch=epoch)

    summary = gradient_tracer.get_summary()

    assert summary["total_epochs"] == 3
    assert summary["total_layers"] == 4
    assert len(summary["layer_names"]) == 4
    assert "layer_stats" in summary
    assert len(summary["layer_stats"]) == 4


def test_empty_gradient_tracer():
    """空のトレーサーのテスト."""
    tracer = GradientTracer()
    summary = tracer.get_summary()
    assert summary == {}


def test_no_gradient(model, gradient_tracer):
    """勾配がない場合のテスト."""
    gradient_tracer.record_gradients(model, epoch=1)

    assert len(gradient_tracer.epochs) == 1
    for layer_name in gradient_tracer.layer_names:
        assert gradient_tracer.gradient_history[layer_name][0] == 0.0


def test_exclude_patterns(model):
    """除外パターンのテスト."""
    tracer = GradientTracer(exclude_patterns=["\\.bias"])

    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))
    output = model(x)
    loss = nn.functional.cross_entropy(output, y)
    loss.backward()

    tracer.record_gradients(model, epoch=1)

    assert len(tracer.layer_names) == 2
    assert "fc1.weight" in tracer.layer_names
    assert "fc2.weight" in tracer.layer_names
    assert "fc1.bias" not in tracer.layer_names
    assert "fc2.bias" not in tracer.layer_names


def test_aggregation_methods(model):
    """集約方法ごとに期待値で集約結果を検証."""
    test_cases = [
        ("median", [1.0, 2.0, 3.0, 4.0], 2.5),
        ("mean", [1.0, 2.0, 3.0, 4.0], 2.5),
        ("max", [1.0, 2.0, 3.0, 4.0], 4.0),
        ("rms", [3.0, 4.0], (12.5) ** 0.5),
    ]

    for method, values, expected in test_cases:
        tracer = GradientTracer(group_by_block=False, aggregation_method=method)
        actual = tracer._aggregate_gradients(values)
        assert actual == pytest.approx(expected, rel=1e-6)


def test_group_by_block():
    """ブロック単位のグループ化テスト."""

    class ResNetLikeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7)
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
            )
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(64, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.layer1(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            return self.fc(x)

    model = ResNetLikeModel()

    tracer = GradientTracer(exclude_patterns=["fc\\.", "\\.bias"], group_by_block=True)

    x = torch.randn(2, 3, 32, 32)
    y = torch.randint(0, 10, (2,))
    output = model(x)
    loss = nn.functional.cross_entropy(output, y)
    loss.backward()

    tracer.record_gradients(model, epoch=1)

    assert "layer1" in tracer.layer_names
    assert "conv1.weight" in tracer.layer_names
    assert "bn1.weight" in tracer.layer_names
    assert not any("layer1.0" in name for name in tracer.layer_names)
