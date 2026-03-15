"""visualize_gradient.py のテスト.

実際の CSV ファイルを書き出して読み込み, PNG 出力を検証する古典派テスト.
"""

from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # テスト環境で GUI バックエンドを使わない

from pochitrain.cli.visualize_gradient import (
    get_method_labels,
    load_gradient_trace,
    plot_heatmap,
    plot_snapshots,
    plot_statistics,
    plot_timeline,
    resolve_config_path,
)


def _write_gradient_csv(path: Path, epochs: int = 5, layers: int = 3) -> Path:
    """テスト用の勾配トレース CSV を書き出す."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "epoch," + ",".join(f"layer{i}" for i in range(layers))
    lines = [header]
    for e in range(1, epochs + 1):
        values = ",".join(str(round(np.random.rand() * 0.1, 6)) for _ in range(layers))
        lines.append(f"{e},{values}")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# --- CSV 読み込みのテスト ---


class TestLoadGradientTrace:
    """load_gradient_trace のテスト."""

    def test_loads_valid_csv(self, tmp_path: Path):
        """正常な CSV ファイルを正しく読み込む."""
        csv_path = _write_gradient_csv(
            tmp_path / "gradient_trace.csv", epochs=3, layers=2
        )

        epochs, layer_names, grad_matrix, method = load_gradient_trace(csv_path)

        assert len(epochs) == 3
        assert layer_names == ["layer0", "layer1"]
        assert grad_matrix.shape == (2, 3)  # layers x epochs
        assert method == "median"  # config がない場合のデフォルト

    def test_epoch_values_correct(self, tmp_path: Path):
        """エポック番号が正しく読み込まれる."""
        csv_path = _write_gradient_csv(tmp_path / "trace.csv", epochs=5)

        epochs, _, _, _ = load_gradient_trace(csv_path)

        np.testing.assert_array_equal(epochs, [1, 2, 3, 4, 5])

    def test_gradient_values_are_float(self, tmp_path: Path):
        """勾配値が float 型で読み込まれる."""
        csv_path = _write_gradient_csv(tmp_path / "trace.csv")

        _, _, grad_matrix, _ = load_gradient_trace(csv_path)

        assert grad_matrix.dtype == np.float64


class TestLoadGradientTraceInvalidCsv:
    """不正な CSV に対するエラーハンドリングのテスト."""

    def test_missing_epoch_column_raises(self, tmp_path: Path):
        """epoch 列がない CSV で KeyError."""
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text("col_a,col_b\n1.0,2.0\n", encoding="utf-8")

        with pytest.raises(KeyError):
            load_gradient_trace(csv_path)

    def test_empty_csv_raises(self, tmp_path: Path):
        """空の CSV でエラー."""
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("", encoding="utf-8")

        with pytest.raises(Exception):
            load_gradient_trace(csv_path)


# --- PNG 出力のテスト ---


class TestPlotTimeline:
    """plot_timeline のテスト."""

    def test_generates_png_files(self, tmp_path: Path):
        """タイムライン PNG ファイルが生成される."""
        epochs = np.array([1, 2, 3, 4, 5])
        layer_names = ["layer0", "layer1", "layer2", "layer3"]
        grad_matrix = np.random.rand(4, 5) * 0.1 + 0.001  # 正値を保証 (log scale)

        plot_timeline(epochs, layer_names, grad_matrix, tmp_path)

        assert (tmp_path / "gradient_trace_timeline_all.png").exists()
        # 4層 > 3 なので early/late も生成される
        assert (tmp_path / "gradient_trace_timeline_early.png").exists()
        assert (tmp_path / "gradient_trace_timeline_late.png").exists()

    def test_few_layers_skips_split(self, tmp_path: Path):
        """3層以下では early/late が生成されない."""
        epochs = np.array([1, 2, 3])
        layer_names = ["layer0", "layer1"]
        grad_matrix = np.random.rand(2, 3) * 0.1 + 0.001

        plot_timeline(epochs, layer_names, grad_matrix, tmp_path)

        assert (tmp_path / "gradient_trace_timeline_all.png").exists()
        assert not (tmp_path / "gradient_trace_timeline_early.png").exists()
        assert not (tmp_path / "gradient_trace_timeline_late.png").exists()


class TestPlotHeatmap:
    """plot_heatmap のテスト."""

    def test_generates_heatmap_png(self, tmp_path: Path):
        """ヒートマップ PNG が生成される."""
        epochs = np.array([1, 2, 3])
        layer_names = ["layer0", "layer1"]
        grad_matrix = np.random.rand(2, 3)

        plot_heatmap(epochs, layer_names, grad_matrix, tmp_path)

        assert (tmp_path / "gradient_trace_heatmap.png").exists()


class TestPlotStatistics:
    """plot_statistics のテスト."""

    def test_generates_statistics_pngs(self, tmp_path: Path):
        """統計情報 PNG ファイルが4つ生成される."""
        epochs = np.array([1, 2, 3, 4, 5])
        layer_names = ["layer0", "layer1", "layer2"]
        grad_matrix = np.random.rand(3, 5) * 0.1 + 0.001

        plot_statistics(epochs, layer_names, grad_matrix, tmp_path)

        assert (tmp_path / "gradient_trace_statistics_initial_vs_final.png").exists()
        assert (tmp_path / "gradient_trace_statistics_stability.png").exists()
        assert (tmp_path / "gradient_trace_statistics_max.png").exists()
        assert (tmp_path / "gradient_trace_statistics_min.png").exists()


class TestPlotSnapshots:
    """plot_snapshots のテスト."""

    def test_generates_snapshot_pngs(self, tmp_path: Path):
        """スナップショット PNG ファイルが2つ生成される."""
        epochs = np.array([1, 2, 3, 4, 5])
        layer_names = ["layer0", "layer1"]
        grad_matrix = np.random.rand(2, 5) * 0.1 + 0.001

        plot_snapshots(epochs, layer_names, grad_matrix, tmp_path)

        assert (tmp_path / "gradient_trace_snapshots.png").exists()
        assert (tmp_path / "gradient_trace_snapshots_linear.png").exists()


# --- ユーティリティ関数のテスト ---


class TestResolveConfigPath:
    """resolve_config_path のテスト."""

    def test_finds_config_in_parent_parent(self, tmp_path: Path):
        """CSV の2階層上に config.py があれば返す."""
        config_path = tmp_path / "config.py"
        config_path.write_text("config = {}", encoding="utf-8")
        csv_path = tmp_path / "visualization" / "gradient_trace.csv"
        csv_path.parent.mkdir(parents=True)

        result = resolve_config_path(csv_path)

        assert result == config_path

    def test_falls_back_to_default_config(self, tmp_path: Path):
        """CSV の2階層上に config.py がない場合, デフォルトのフォールバックを試みる."""
        csv_path = tmp_path / "somewhere" / "gradient_trace.csv"
        csv_path.parent.mkdir(parents=True)

        result = resolve_config_path(csv_path)

        # 親に config.py がなければフォールバック候補 (configs/pochi_train_config.py) を試みる
        # プロジェクトルートに存在する場合はそのパスが返り, なければ None
        if result is not None:
            assert result.name == "pochi_train_config.py"
        else:
            assert result is None


class TestGetMethodLabels:
    """get_method_labels のテスト."""

    def test_known_methods(self):
        """既知の集約方法の表示名を返す."""
        assert get_method_labels("median") == ("中央値", "層内中央値")
        assert get_method_labels("mean") == ("平均値", "層内平均値")
        assert get_method_labels("max") == ("最大値", "層内最大値")
        assert get_method_labels("rms") == ("RMS", "層内RMS")

    def test_unknown_method_uses_raw_name(self):
        """未知の方法はそのまま返す."""
        label, inner = get_method_labels("custom")
        assert label == "custom"
        assert inner == "層内custom"
