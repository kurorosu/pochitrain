"""勾配トレースCSVから可視化を生成するスクリプト."""

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 日本語フォント設定
plt.rcParams["font.sans-serif"] = ["MS Gothic", "Yu Gothic", "Meiryo"]
plt.rcParams["axes.unicode_minus"] = False


METHOD_LABELS = {
    "median": "中央値",
    "mean": "平均値",
    "max": "最大値",
    "rms": "RMS",
}


def resolve_config_path(csv_path: Path) -> Optional[Path]:
    """CSVファイルに紐づく設定ファイルの候補パスを取得."""
    candidates = [
        csv_path.parent.parent / "config.py",
        Path("configs/pochi_train_config.py"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def load_config_dict(config_path: Path) -> dict:
    """訓練時と同じ方法で設定ファイルを辞書として読み込む."""
    spec = importlib.util.spec_from_file_location("pochitrain_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"設定ファイルの読み込みに失敗しました: {config_path}")

    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    config = {}
    for key in dir(config_module):
        if key.startswith("_"):
            continue

        value = getattr(config_module, key)
        if callable(value) and not hasattr(value, "transforms"):
            continue

        config[key] = value

    return config


def get_method_labels(aggregation_method: str) -> Tuple[str, str]:
    """集約方法に対応する表示名と層内ラベルを返す."""
    method_label = METHOD_LABELS.get(aggregation_method, aggregation_method)
    inner_label = f"層内{method_label}"
    return method_label, inner_label


def load_gradient_trace(
    csv_path: Path,
) -> Tuple[np.ndarray, List[str], np.ndarray, str]:
    """
    勾配トレースCSVを読み込み.

    Args:
        csv_path (Path): CSVファイルパス

    Returns:
        tuple: (epochs, layer_names, grad_matrix, aggregation_method)
    """
    df = pd.read_csv(csv_path)

    # エポック番号
    epochs = df["epoch"].to_numpy()

    # 層名（epoch列以外）
    layer_names = [col for col in df.columns if col != "epoch"]

    # 勾配ノルムの行列（層 x エポック）
    grad_matrix = df[layer_names].to_numpy(dtype=float).T

    aggregation_method = "median"

    config_path = resolve_config_path(csv_path)
    if config_path is not None:
        try:
            config = load_config_dict(config_path)
        except Exception as exc:  # noqa: BLE001
            print(
                f"警告: 設定ファイルの読み込みに失敗しました ({exc})", file=sys.stderr
            )
        else:
            gradient_cfg = config.get("gradient_tracking_config", {})
            aggregation_method = gradient_cfg.get(
                "aggregation_method", aggregation_method
            )

    return epochs, layer_names, grad_matrix, aggregation_method


def plot_timeline(
    epochs: np.ndarray,
    layer_names: List[str],
    grad_matrix: np.ndarray,
    output_dir: Path,
    aggregation_method: str = "median",
) -> None:
    """
    時系列プロットを生成（1グラフ1画像）.

    Args:
        epochs (np.ndarray): エポック番号
        layer_names (list): 層名リスト
        grad_matrix (np.ndarray): 勾配ノルム行列
        output_dir (Path): 出力ディレクトリ
        aggregation_method (str): 集約方法
    """
    _, inner_label = get_method_labels(aggregation_method)

    def draw_timeline(
        filename: str,
        indices: np.ndarray,
        title: str,
        highlight_threshold: Optional[float] = None,
        progress_idx: int = 0,
    ) -> None:
        fig, ax = plt.subplots(figsize=(12, 6))

        for idx in indices:
            ax.plot(
                epochs,
                grad_matrix[idx],
                marker="o",
                markersize=3,
                label=layer_names[idx],
                linewidth=1.5,
                alpha=0.8,
            )

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(f"Gradient Norm ({inner_label})", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

        if highlight_threshold is not None:
            ax.axhline(
                y=highlight_threshold,
                color="red",
                linestyle="--",
                linewidth=1,
                alpha=0.5,
            )

        plt.tight_layout()
        output_path = output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[{progress_idx}/10] {title} を保存: {output_path}")

    draw_timeline(
        "gradient_trace_timeline_all.png",
        np.arange(len(layer_names)),
        f"層ごとの勾配ノルムの推移 ({inner_label})",
        progress_idx=1,
    )

    if len(layer_names) > 3:
        early_count = min(len(layer_names) // 2, len(layer_names))
        draw_timeline(
            "gradient_trace_timeline_early.png",
            np.arange(early_count),
            f"前半層の勾配ノルム ({inner_label})",
            highlight_threshold=0.001,
            progress_idx=2,
        )

        late_indices = np.arange(len(layer_names) // 2, len(layer_names))
        draw_timeline(
            "gradient_trace_timeline_late.png",
            late_indices,
            f"後半層の勾配ノルム ({inner_label})",
            progress_idx=3,
        )


def plot_heatmap(
    epochs: np.ndarray,
    layer_names: List[str],
    grad_matrix: np.ndarray,
    output_dir: Path,
    aggregation_method: str = "median",
) -> None:
    """
    ヒートマップを生成.

    Args:
        epochs (np.ndarray): エポック番号
        layer_names (list): 層名リスト
        grad_matrix (np.ndarray): 勾配ノルム行列
        output_dir (Path): 出力ディレクトリ
        aggregation_method (str): 集約方法
    """
    _, inner_label = get_method_labels(aggregation_method)
    fig, ax = plt.subplots(figsize=(14, max(6, len(layer_names) * 0.3)))

    # ヒートマップを描画
    im = ax.imshow(grad_matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    # カラーバー
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f"Gradient Norm ({inner_label})", fontsize=11)

    # 軸設定
    n_epochs = len(epochs)
    step = max(1, n_epochs // 20)  # 最大20個のティック
    tick_positions = list(range(0, n_epochs, step))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(epochs[tick_positions])
    ax.set_yticks(range(len(layer_names)))
    ax.set_yticklabels(layer_names, fontsize=8)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title(
        f"層×エポックの勾配ノルム ヒートマップ ({inner_label})",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    output_path = output_dir / "gradient_trace_heatmap.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[4/10] ヒートマップを保存: {output_path}")


def plot_statistics(
    epochs: np.ndarray,
    layer_names: List[str],
    grad_matrix: np.ndarray,
    output_dir: Path,
    aggregation_method: str = "median",
) -> None:
    """
    統計情報グラフを生成（1グラフ1画像）.

    Args:
        epochs (np.ndarray): エポック番号
        layer_names (list): 層名リスト
        grad_matrix (np.ndarray): 勾配ノルム行列
        output_dir (Path): 出力ディレクトリ
        aggregation_method (str): 集約方法
    """
    method_label, inner_label = get_method_labels(aggregation_method)
    # (1) 初期 vs 最終エポック
    fig, ax = plt.subplots(figsize=(12, 6))
    initial_window = min(5, len(epochs) // 10) if len(epochs) > 10 else 1
    final_window = min(5, len(epochs) // 10) if len(epochs) > 10 else 1

    initial_grads = grad_matrix[:, :initial_window].mean(axis=1)
    final_grads = grad_matrix[:, -final_window:].mean(axis=1)

    x = np.arange(len(layer_names))
    width = 0.35

    ax.bar(
        x - width / 2,
        initial_grads,
        width,
        label=f"初期 (1-{initial_window})",
        alpha=0.8,
    )
    ax.bar(
        x + width / 2,
        final_grads,
        width,
        label=f"最終 ({len(epochs)-final_window+1}-{len(epochs)})",
        alpha=0.8,
    )

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Average Gradient Norm (epoch方向平均値)", fontsize=12)
    ax.set_title(
        f"初期 vs 最終エポックの勾配比較 ({inner_label})",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = output_dir / "gradient_trace_statistics_initial_vs_final.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[5/10] 統計情報（初期 vs 最終）を保存: {output_path}")

    # (2) 標準偏差（安定性）
    fig, ax = plt.subplots(figsize=(12, 6))
    grad_stds = grad_matrix.std(axis=1)
    threshold = np.median(grad_stds) * 2
    colors = ["red" if std > threshold else "green" for std in grad_stds]
    indices = np.arange(len(layer_names))

    ax.bar(indices, grad_stds, color=colors, alpha=0.7)
    ax.axhline(
        y=threshold, color="red", linestyle="--", linewidth=1, label="不安定閾値"
    )

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Gradient Norm Std Dev (epoch方向標準偏差)", fontsize=12)
    ax.set_title(
        f"勾配の変動（安定性の指標） ({inner_label})", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(indices)
    ax.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = output_dir / "gradient_trace_statistics_stability.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[6/10] 統計情報（安定性）を保存: {output_path}")

    # (3) 最大値（爆発検出）
    fig, ax = plt.subplots(figsize=(12, 6))
    grad_maxs = grad_matrix.max(axis=1)
    threshold_max = np.median(grad_maxs) * 3
    colors = ["red" if maxval > threshold_max else "blue" for maxval in grad_maxs]

    ax.bar(indices, grad_maxs, color=colors, alpha=0.7)
    ax.axhline(
        y=threshold_max, color="red", linestyle="--", linewidth=1, label="爆発閾値"
    )

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Max Gradient Norm (epoch方向最大値)", fontsize=12)
    ax.set_title(
        f"勾配の最大値（爆発の検出） ({inner_label})", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(indices)
    ax.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = output_dir / "gradient_trace_statistics_max.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[7/10] 統計情報（最大値）を保存: {output_path}")

    # (4) 最小値（消失検出）
    fig, ax = plt.subplots(figsize=(12, 6))
    grad_mins = grad_matrix.min(axis=1)
    threshold_min = 0.0001
    colors = ["red" if minval < threshold_min else "blue" for minval in grad_mins]

    ax.bar(indices, grad_mins, color=colors, alpha=0.7)
    ax.axhline(
        y=threshold_min, color="red", linestyle="--", linewidth=1, label="消失閾値"
    )

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Min Gradient Norm (epoch方向最小値)", fontsize=12)
    ax.set_title(
        f"勾配の最小値（消失の検出） ({inner_label})", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(indices)
    ax.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = output_dir / "gradient_trace_statistics_min.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[8/10] 統計情報（最小値）を保存: {output_path}")


def plot_snapshots(
    epochs: np.ndarray,
    layer_names: List[str],
    grad_matrix: np.ndarray,
    output_dir: Path,
    aggregation_method: str = "median",
) -> None:
    """
    エポックスナップショット（対数スケール）を生成.

    Args:
        epochs (np.ndarray): エポック番号
        layer_names (list): 層名リスト
        grad_matrix (np.ndarray): 勾配ノルム行列
        output_dir (Path): 出力ディレクトリ
        aggregation_method (str): 集約方法
    """
    _, inner_label = get_method_labels(aggregation_method)
    fig, ax = plt.subplots(figsize=(12, 6))
    x_positions = np.arange(len(layer_names))

    # スナップショットを取るエポック（最大6個）
    n_snapshots = min(6, len(epochs))
    snapshot_indices = (
        np.linspace(0, len(epochs) - 1, n_snapshots, dtype=int)
        if n_snapshots > 1
        else np.array([0])
    )

    colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(snapshot_indices)))

    for i, epoch_idx in enumerate(snapshot_indices):
        grad_values = grad_matrix[:, epoch_idx]
        ax.plot(
            x_positions,
            grad_values,
            marker="o",
            markersize=8,
            label=f"Epoch {epochs[epoch_idx]}",
            linewidth=2,
            alpha=0.8,
            color=colors[i],
        )

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel(f"Gradient Norm ({inner_label})", fontsize=12)
    ax.set_title(
        f"エポックごとの勾配プロファイル ({inner_label})",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")  # 対数スケール
    ax.set_xticks(x_positions)
    ax.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=10)

    plt.tight_layout()
    output_path = output_dir / "gradient_trace_snapshots.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[9/10] スナップショットを保存: {output_path}")

    # 線形スケール版も生成
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, epoch_idx in enumerate(snapshot_indices):
        grad_values = grad_matrix[:, epoch_idx]
        ax.plot(
            x_positions,
            grad_values,
            marker="o",
            markersize=8,
            label=f"Epoch {epochs[epoch_idx]}",
            linewidth=2,
            alpha=0.8,
            color=colors[i],
        )

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel(f"Gradient Norm ({inner_label})", fontsize=12)
    ax.set_title(
        f"エポックごとの勾配プロファイル（線形スケール） ({inner_label})",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=10)

    plt.tight_layout()
    output_path = output_dir / "gradient_trace_snapshots_linear.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[10/10] スナップショット（線形）を保存: {output_path}")


def main() -> None:
    """メイン処理."""
    parser = argparse.ArgumentParser(
        description="勾配トレースCSVから可視化グラフを生成"
    )
    parser.add_argument("csv_path", type=str, help="勾配トレースCSVファイルのパス")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="出力ディレクトリ（デフォルト: CSVと同じディレクトリ）",
    )

    args = parser.parse_args()

    # パスの準備
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"エラー: CSVファイルが見つかりません: {csv_path}")
        sys.exit(1)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = csv_path.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"勾配トレースCSVを読み込み: {csv_path}")

    # データ読み込み
    epochs, layer_names, grad_matrix, aggregation_method = load_gradient_trace(csv_path)

    print(f"  - エポック数: {len(epochs)}")
    print(f"  - 層数: {len(layer_names)}")
    print(f"  - 集約方法: {aggregation_method}")
    print()

    # 可視化生成
    print("可視化を生成中...")
    plot_timeline(epochs, layer_names, grad_matrix, output_dir, aggregation_method)
    plot_heatmap(epochs, layer_names, grad_matrix, output_dir, aggregation_method)
    plot_statistics(epochs, layer_names, grad_matrix, output_dir, aggregation_method)
    plot_snapshots(epochs, layer_names, grad_matrix, output_dir, aggregation_method)

    print()
    print("[完了] すべての可視化が完了しました！")
    print(f"生成された画像（{output_dir}）:")


if __name__ == "__main__":
    main()
