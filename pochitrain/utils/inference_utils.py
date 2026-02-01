"""推論CLI共通ユーティリティ.

PyTorch, ONNX, TensorRT推論CLIで共通して使用する処理を提供.
"""

import csv
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from pochitrain.logging import LoggerManager

logger: logging.Logger = LoggerManager().get_logger(__name__)


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
    # 数値安定性のためmax減算してからsoftmax
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
    基本的なNumPy操作のみで混同行列を計算します.

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
    import matplotlib.pyplot as plt
    import matplotlib_fontja  # noqa: F401

    matplotlib.use("Agg")

    # デフォルト設定
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

    # 設定をマージ（cm_configが指定されていれば優先）
    config = default_config.copy()
    if cm_config:
        config.update(cm_config)

    # 混同行列を計算
    cm = compute_confusion_matrix(predicted_labels, true_labels, len(class_names))

    # プロット作成
    fig, ax = plt.subplots(figsize=config["figsize"])

    # ヒートマップを描画
    cmap_value: str = str(config["cmap"])
    ax.imshow(cm, interpolation="nearest", cmap=cmap_value)

    # ラベル設定
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    # ラベルとタイトル
    xlabel: str = str(config["xlabel"])
    ylabel: str = str(config["ylabel"])
    title: str = str(config["title"])
    ax.set_xlabel(xlabel, fontsize=config["label_fontsize"])
    ax.set_ylabel(ylabel, fontsize=config["label_fontsize"])
    ax.set_title(title, fontsize=config["title_fontsize"])

    # 各セルに数値を表示
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

    # レイアウト調整
    plt.tight_layout()

    # ファイル保存
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
    # models フォルダ -> work_dir フォルダ -> config.py
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
    # models フォルダ -> work_dir フォルダ -> inference_results
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
        logger.info(f"設定ファイルを読み込み: {config_path}")
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

    logger.info(f"結果を保存: {csv_path}")
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
        f.write(f"サンプル数: {num_samples}\n")
        f.write(f"精度: {accuracy:.2f}%\n")
        f.write(f"平均推論時間: {avg_time_per_image:.2f} ms/image\n")
        f.write(f"スループット: {throughput:.1f} images/sec\n")
        f.write(
            f"計測サンプル数: {total_samples} (ウォームアップ除外: {warmup_samples})\n"
        )

        if extra_info:
            for key, value in extra_info.items():
                f.write(f"{key}: {value}\n")

    logger.info(f"サマリーを保存: {summary_path}")
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

    # クラス別メトリクス
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
) -> None:
    """推論結果をログに出力する.

    Args:
        num_samples: 総サンプル数
        correct: 正解数
        avg_time_per_image: 平均推論時間 (ms/image)
        total_samples: 計測サンプル数
        warmup_samples: ウォームアップ除外サンプル数
    """
    accuracy = (correct / num_samples) * 100 if num_samples > 0 else 0.0
    throughput = 1000 / avg_time_per_image if avg_time_per_image > 0 else 0

    logger.info("推論完了")
    logger.info(f"精度: {correct}/{num_samples} ({accuracy:.2f}%)")
    logger.info(
        f"平均推論時間: {avg_time_per_image:.2f} ms/image "
        f"(計測: {total_samples}枚, ウォームアップ除外: {warmup_samples}枚)"
    )
    logger.info(f"スループット: {throughput:.1f} images/sec")
