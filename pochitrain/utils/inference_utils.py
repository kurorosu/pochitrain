"""推論CLI共通ユーティリティ.

PyTorch, ONNX, TensorRT推論CLIで共通して使用する処理を提供.
"""

import csv
import importlib
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from pochitrain.logging import LoggerManager

logger: logging.Logger = LoggerManager().get_logger(__name__)
_MATPLOTLIB_FONTJA_WARNING_EMITTED = False


def _import_matplotlib_fontja_if_available() -> None:
    """matplotlib_fontja を利用可能な場合のみ読み込む.

    Note:
        Jetson 等の環境では matplotlib_fontja が未導入でも推論結果出力を継続する.
        未導入時は既定フォントで描画する.
    """
    global _MATPLOTLIB_FONTJA_WARNING_EMITTED

    try:
        importlib.import_module("matplotlib_fontja")
    except ImportError:
        if not _MATPLOTLIB_FONTJA_WARNING_EMITTED:
            logger.warning(
                "matplotlib_fontja が見つからないため, "
                "既定フォントで混同行列を描画します."
            )
            _MATPLOTLIB_FONTJA_WARNING_EMITTED = True


def post_process_logits(
    logits: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """ロジットからsoftmax + argmax + confidence抽出を行う共通後処理.

    NumPy配列のロジットを受け取り, softmaxで確率に変換後,
    予測クラスと信頼度を返す.

    Args:
        logits: モデル出力のロジット (batch_size, num_classes)

    Returns:
        (predicted, confidence) のタプル.
        predicted: 予測クラスインデックス (batch_size,)
        confidence: 最大確率値 (batch_size,)
    """
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    predicted = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)

    return predicted, confidence


def compute_confusion_matrix(
    predicted_labels: List[int],
    true_labels: List[int],
    num_classes: int,
) -> np.ndarray:
    """NumPyベースの混同行列計算.

    sklearn.metrics.confusion_matrixを使用せず,
    基本的なNumPy操作のみで混同行列を計算する.
    推論CLIはONNX/TRT/PyTorchの出力を最終的にlist[int]へ正規化して扱うため,
    本関数はTorch非依存で実装している.
    訓練ループ内のTorch Tensor向け実装は ``pochitrain.training.evaluator`` に分離している.

    Args:
        predicted_labels: 予測ラベルのリスト
        true_labels: 正解ラベルのリスト
        num_classes: クラス数

    Returns:
        混同行列 (shape: [num_classes, num_classes])
        行が正解ラベル, 列が予測ラベルに対応.
        confusion_matrix[i, j] = 正解がi, 予測がjの個数
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(true_labels, predicted_labels):
        cm[t, p] += 1
    return cm


def save_confusion_matrix_image(
    predicted_labels: List[int],
    true_labels: List[int],
    class_names: List[str],
    output_dir: Path,
    filename: str = "confusion_matrix.png",
    cm_config: Optional[Dict[str, Any]] = None,
) -> Path:
    """混同行列の画像を保存.

    Args:
        predicted_labels: 予測ラベルのリスト
        true_labels: 正解ラベルのリスト
        class_names: クラス名のリスト
        output_dir: 出力ディレクトリ
        filename: 保存ファイル名
        cm_config: 混同行列可視化設定

    Returns:
        保存されたファイルのパス
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _import_matplotlib_fontja_if_available()

    default_config: Dict[str, Any] = {
        "title": "Confusion Matrix",
        "xlabel": "Predicted Label",
        "ylabel": "True Label",
        "fontsize": 14,
        "title_fontsize": 16,
        "label_fontsize": 12,
        "figsize": (8, 6),
        "cmap": "Blues",
    }

    config = default_config.copy()
    if cm_config:
        config.update(cm_config)

    cm = compute_confusion_matrix(predicted_labels, true_labels, len(class_names))

    fig, ax = plt.subplots(figsize=config["figsize"])

    cmap_value: str = str(config["cmap"])
    ax.imshow(cm, interpolation="nearest", cmap=cmap_value)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    xlabel: str = str(config["xlabel"])
    ylabel: str = str(config["ylabel"])
    title: str = str(config["title"])
    ax.set_xlabel(xlabel, fontsize=config["label_fontsize"])
    ax.set_ylabel(ylabel, fontsize=config["label_fontsize"])
    ax.set_title(title, fontsize=config["title_fontsize"])

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

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.debug(f"混同行列画像保存: {output_path}")
    return output_path


def auto_detect_config_path(model_path: Path) -> Path:
    """モデルパスからconfig.pyを自動検出する.

    モデルパスの親ディレクトリ（models/）の更に親ディレクトリ（work_dir/）にある
    config.pyを検出する.

    Args:
        model_path: モデルファイルパス (例: work_dirs/20260126_001/models/model.pth)

    Returns:
        検出されたconfig.pyのパス (例: work_dirs/20260126_001/config.py)
    """
    work_dir = model_path.parent.parent
    return work_dir / "config.py"


def get_default_output_base_dir(model_path: Path) -> Path:
    """モデルパスからデフォルト出力先のベースディレクトリを返す.

    モデルパスの親ディレクトリ（models/）と同階層のinference_results/を返す.
    ワークスペースの作成は行わない.

    Args:
        model_path: モデルファイルパス (例: work_dirs/20260126_001/models/model.pth)

    Returns:
        ベースディレクトリパス (例: work_dirs/20260126_001/inference_results/)
    """
    work_dir = model_path.parent.parent
    return work_dir / "inference_results"


def validate_model_path(model_path: Path) -> None:
    """モデルパスの存在を検証する.

    Args:
        model_path: モデルファイルパス

    Raises:
        SystemExit: モデルファイルが存在しない場合
    """
    if not model_path.exists():
        logger.error(f"モデルファイルが見つかりません: {model_path}")
        sys.exit(1)


def validate_data_path(data_path: Path) -> None:
    """データパスの存在を検証する.

    Args:
        data_path: データディレクトリパス

    Raises:
        SystemExit: データディレクトリが存在しない場合
    """
    if not data_path.exists():
        logger.error(f"データディレクトリが見つかりません: {data_path}")
        sys.exit(1)


def load_config_auto(model_path: Path) -> Dict[str, Any]:
    """モデルパスからconfigを自動検出して読み込む.

    Args:
        model_path: モデルファイルパス

    Returns:
        読み込んだconfig辞書

    Raises:
        SystemExit: configファイルが見つからない、または読み込めない場合
    """
    config_path = auto_detect_config_path(model_path)
    if not config_path.exists():
        logger.error(f"設定ファイルが見つかりません: {config_path}")
        logger.error("モデルパスと同じwork_dir内にconfig.pyが必要です")
        sys.exit(1)

    try:
        from pochitrain.utils.config_loader import ConfigLoader

        config = ConfigLoader.load_config(str(config_path))
        logger.debug(f"設定ファイルを読み込み: {config_path}")
        return config
    except Exception as e:
        logger.error(f"設定ファイル読み込みエラー: {e}")
        sys.exit(1)


def write_inference_csv(
    output_dir: Path,
    image_paths: List[str],
    predictions: List[int],
    true_labels: List[int],
    confidences: List[float],
    class_names: List[str],
    filename: str = "inference_results.csv",
) -> Path:
    """推論結果をCSVに出力する.

    Args:
        output_dir: 出力ディレクトリ
        image_paths: 画像パスリスト
        predictions: 予測ラベルリスト
        true_labels: 正解ラベルリスト
        confidences: 信頼度リスト
        class_names: クラス名リスト
        filename: 出力ファイル名

    Returns:
        出力したCSVファイルのパス
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / filename

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image_path",
                "predicted",
                "predicted_class",
                "true",
                "true_class",
                "confidence",
                "correct",
            ]
        )
        for path, pred, true, conf in zip(
            image_paths, predictions, true_labels, confidences
        ):
            writer.writerow(
                [
                    path,
                    pred,
                    class_names[pred],
                    true,
                    class_names[true],
                    f"{conf:.4f}",
                    pred == true,
                ]
            )

    logger.debug(f"詳細結果: {csv_path}")
    return csv_path


def write_inference_summary(
    output_dir: Path,
    model_path: Path,
    data_path: Path,
    num_samples: int,
    accuracy: float,
    avg_time_per_image: float,
    total_samples: int,
    warmup_samples: int,
    avg_total_time_per_image: Optional[float] = None,
    input_size: Optional[Tuple[int, int, int]] = None,
    filename: str = "inference_summary.txt",
    extra_info: Optional[Dict[str, Any]] = None,
) -> Path:
    """推論サマリーをファイルに出力する.

    Args:
        output_dir: 出力ディレクトリ
        model_path: モデルファイルパス
        data_path: データディレクトリパス
        num_samples: 総サンプル数
        accuracy: 精度 (%)
        avg_time_per_image: 平均推論時間 (ms/image)
        total_samples: 計測サンプル数
        warmup_samples: ウォームアップ除外サンプル数
        avg_total_time_per_image: 平均全処理時間 (ms/image, I/O・転送等すべて含む)
        input_size: 入力サイズ (C, H, W)
        filename: 出力ファイル名
        extra_info: 追加情報（任意）

    Returns:
        出力したサマリーファイルのパス
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / filename

    throughput = 1000 / avg_time_per_image if avg_time_per_image > 0 else 0

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"モデル: {model_path}\n")
        f.write(f"データ: {data_path}\n")
        if input_size:
            f.write(f"入力解像度: {input_size[2]}x{input_size[1]} (WxH)\n")
            f.write(f"入力チャンネル: {input_size[0]}\n")
        f.write(f"サンプル数: {num_samples}\n")
        f.write(f"精度: {accuracy:.2f}%\n")
        f.write(f"平均推論時間: {avg_time_per_image:.2f} ms/image (純粋推論のみ)\n")
        f.write(f"スループット: {throughput:.1f} images/sec (純粋推論ベース)\n")

        if avg_total_time_per_image is not None:
            total_throughput = (
                1000 / avg_total_time_per_image if avg_total_time_per_image > 0 else 0
            )
            f.write(
                f"平均全処理時間: {avg_total_time_per_image:.2f} ms/image (End-to-End)\n"
            )
            f.write(
                f"スループット: {total_throughput:.1f} images/sec (実効性能ベース)\n"
            )

        f.write(
            f"計測サンプル数: {total_samples} (ウォームアップ除外: {warmup_samples})\n"
        )

        if extra_info:
            for key, value in extra_info.items():
                f.write(f"{key}: {value}\n")

    logger.debug(f"サマリー: {summary_path}")
    return summary_path


def save_classification_report(
    predicted_labels: List[int],
    true_labels: List[int],
    class_names: List[str],
    output_dir: Path,
    filename: str = "classification_report.csv",
) -> Path:
    """クラス別精度レポートをCSVに保存.

    混同行列からPrecision, Recall, F1-scoreをクラスごとに計算し,
    macro avg, weighted avgとともにCSV出力する.

    Args:
        predicted_labels: 予測ラベルのリスト
        true_labels: 正解ラベルのリスト
        class_names: クラス名のリスト
        output_dir: 出力ディレクトリ
        filename: 出力ファイル名

    Returns:
        保存されたCSVファイルのパス
    """
    num_classes = len(class_names)
    labels = list(range(num_classes))

    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predicted_labels, labels=labels, zero_division=0
    )

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            true_labels, predicted_labels, average="weighted", zero_division=0
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / filename

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1-score", "support"])

        for i, name in enumerate(class_names):
            writer.writerow(
                [
                    name,
                    f"{precision[i]:.4f}",
                    f"{recall[i]:.4f}",
                    f"{f1[i]:.4f}",
                    int(support[i]),
                ]
            )

        total_support = int(sum(support))
        writer.writerow(
            [
                "macro avg",
                f"{precision_macro:.4f}",
                f"{recall_macro:.4f}",
                f"{f1_macro:.4f}",
                total_support,
            ]
        )
        writer.writerow(
            [
                "weighted avg",
                f"{precision_weighted:.4f}",
                f"{recall_weighted:.4f}",
                f"{f1_weighted:.4f}",
                total_support,
            ]
        )

    logger.debug(f"クラス別精度レポート保存: {csv_path}")
    return csv_path


def log_inference_result(
    num_samples: int,
    correct: int,
    avg_time_per_image: float,
    total_samples: int,
    warmup_samples: int,
    avg_total_time_per_image: Optional[float] = None,
    input_size: Optional[Tuple[int, int, int]] = None,
) -> None:
    """推論結果をログに出力する.

    Args:
        num_samples: 総サンプル数
        correct: 正解数
        avg_time_per_image: 平均推論時間 (ms/image)
        total_samples: 計測サンプル数
        warmup_samples: ウォームアップ除外サンプル数
        avg_total_time_per_image: 平均全処理時間 (ms/image, I/O・転送等すべて含む)
        input_size: 入力サイズ (C, H, W)
    """
    accuracy = (correct / num_samples) * 100 if num_samples > 0 else 0.0
    throughput = 1000 / avg_time_per_image if avg_time_per_image > 0 else 0

    logger.info(f"推論画像枚数: {num_samples}枚")
    if input_size:
        logger.info(
            f"入力解像度: {input_size[2]}x{input_size[1]} (WxH), チャンネル数: {input_size[0]}"
        )
    logger.info(f"精度: {accuracy:.2f}%")

    logger.info(
        f"平均推論時間: {avg_time_per_image:.2f} ms/image, 計測範囲: 純粋推論のみ (転送・I/O除外)"
    )
    logger.info(
        f"スループット: {throughput:.1f} images/sec, 計測範囲: 純粋推論のみ (転送・I/O除外)"
    )

    if avg_total_time_per_image is not None:
        total_throughput = (
            1000 / avg_total_time_per_image if avg_total_time_per_image > 0 else 0
        )
        logger.info(
            f"平均全処理時間: {avg_total_time_per_image:.2f} ms/image, 計測範囲: 全処理 (I/O・前処理・転送・推論・後処理込み)"
        )
        logger.info(
            f"スループット: {total_throughput:.1f} images/sec, 計測範囲: 全処理 (I/O・前処理・転送・推論・後処理込み)"
        )

    logger.info(f"計測詳細: {total_samples}枚, ウォームアップ除外: {warmup_samples}枚")
