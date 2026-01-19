#!/usr/bin/env python3
"""
ONNXモデルを使用した推論スクリプト.

使用例:
    python tools/infer_onnx.py model.onnx --data data/val --config config.py
    python tools/infer_onnx.py model.onnx --data data/val --input-size 512 512
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

# onnxruntimeの確認
try:
    import onnxruntime as ort
except ImportError:
    print("エラー: onnxruntimeパッケージがインストールされていません")
    print("インストール: pip install onnxruntime または pip install onnxruntime-gpu")
    sys.exit(1)

from pochitrain.utils import ConfigLoader


def create_onnx_session(
    model_path: Path,
    use_gpu: bool = False,
) -> ort.InferenceSession:
    """ONNXセッションを作成."""
    providers = []
    if use_gpu:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    session = ort.InferenceSession(str(model_path), providers=providers)
    return session


def run_inference(
    session: ort.InferenceSession,
    images: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """推論を実行."""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # 推論実行
    outputs = session.run([output_name], {input_name: images})
    logits = outputs[0]

    # softmaxで確率に変換
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # 予測クラスと信頼度
    predicted = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)

    return predicted, confidence


def main() -> None:
    """メイン関数."""
    parser = argparse.ArgumentParser(
        description="ONNXモデルを使用した推論",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本的な推論
  python tools/infer_onnx.py model.onnx --data data/val --config config.py

  # 入力サイズを直接指定
  python tools/infer_onnx.py model.onnx --data data/val --input-size 512 512

  # GPU使用
  python tools/infer_onnx.py model.onnx --data data/val --config config.py --gpu
        """,
    )

    parser.add_argument("model_path", help="ONNXモデルファイルパス")
    parser.add_argument(
        "--data",
        required=True,
        help="推論データディレクトリ",
    )
    parser.add_argument(
        "--config",
        "-c",
        help="設定ファイルパス（変換設定を取得）",
    )
    parser.add_argument(
        "--input-size",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        help="入力画像サイズ（configがない場合は必須）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="バッチサイズ (default: 1)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="GPUを使用（onnxruntime-gpuが必要）",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="結果出力ディレクトリ",
    )

    args = parser.parse_args()

    # モデルパスの確認
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"エラー: モデルファイルが見つかりません: {model_path}")
        sys.exit(1)

    # データパスの確認
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"エラー: データディレクトリが見つかりません: {data_path}")
        sys.exit(1)

    # 設定ファイルの読み込み
    config = None
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            try:
                config = ConfigLoader.load_config(str(config_path))
                print(f"設定ファイルを読み込み: {config_path}")
            except Exception as e:
                print(f"警告: 設定ファイルの読み込みに失敗: {e}")

    # 入力サイズの決定
    if args.input_size:
        input_size = (args.input_size[0], args.input_size[1])
    elif config and "val_transform" in config:
        # transformからサイズを推測（Resizeがあれば）
        input_size = (224, 224)  # デフォルト
        print("警告: 入力サイズを224x224と仮定しています")
    else:
        print("エラー: --input-size を指定するか、--config を使用してください")
        sys.exit(1)

    # バッチサイズ
    batch_size = args.batch_size

    print(f"モデル: {model_path}")
    print(f"データ: {data_path}")
    print(f"入力サイズ: {input_size[0]}x{input_size[1]}")
    print(f"バッチサイズ: {batch_size}")
    print(f"GPU使用: {args.gpu}")

    # pochitrainのデータセットを使用
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from pochitrain.pochi_dataset import PochiImageDataset, get_basic_transforms

    # データセット作成
    transform = get_basic_transforms(image_size=input_size[0], is_training=False)
    dataset = PochiImageDataset(str(data_path), transform=transform)

    print(f"データセット: {len(dataset)}枚")
    print(f"クラス: {dataset.get_classes()}")

    # ONNXセッション作成
    print("ONNXセッションを作成中...")
    session = create_onnx_session(model_path, use_gpu=args.gpu)

    # 使用中のプロバイダーを表示
    providers = session.get_providers()
    print(f"実行プロバイダー: {providers}")

    # 推論実行
    print("推論を開始...")
    all_predictions: List[int] = []
    all_confidences: List[float] = []
    all_true_labels: List[int] = []
    total_inference_time = 0.0
    total_samples = 0

    # バッチ処理
    num_batches = (len(dataset) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(dataset))

        # バッチデータの準備
        batch_images = []
        batch_labels = []
        for i in range(start_idx, end_idx):
            image, label = dataset[i]
            batch_images.append(image.numpy())
            batch_labels.append(label)

        # NumPy配列に変換
        images_np = np.stack(batch_images).astype(np.float32)

        # 推論時間計測
        start_time = time.perf_counter()
        predicted, confidence = run_inference(session, images_np)
        inference_time = (time.perf_counter() - start_time) * 1000  # ms

        total_inference_time += inference_time
        total_samples += len(batch_images)

        all_predictions.extend(predicted.tolist())
        all_confidences.extend(confidence.tolist())
        all_true_labels.extend(batch_labels)

    # 精度計算
    correct = sum(p == t for p, t in zip(all_predictions, all_true_labels))
    accuracy = (correct / total_samples) * 100 if total_samples > 0 else 0.0
    avg_time_per_image = (
        total_inference_time / total_samples if total_samples > 0 else 0
    )

    print(f"\n推論完了")
    print(f"精度: {correct}/{total_samples} ({accuracy:.2f}%)")
    print(f"平均推論時間: {avg_time_per_image:.2f} ms/image")
    print(f"総推論時間: {total_inference_time:.2f} ms")

    # 結果出力
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # CSV出力
        import csv

        csv_path = output_dir / "onnx_inference_results.csv"
        class_names = dataset.get_classes()
        image_paths = dataset.get_file_paths()

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
            for i, (path, pred, true, conf) in enumerate(
                zip(image_paths, all_predictions, all_true_labels, all_confidences)
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

        print(f"結果を保存: {csv_path}")

        # サマリー出力
        summary_path = output_dir / "onnx_inference_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"モデル: {model_path}\n")
            f.write(f"データ: {data_path}\n")
            f.write(f"サンプル数: {total_samples}\n")
            f.write(f"精度: {accuracy:.2f}%\n")
            f.write(f"平均推論時間: {avg_time_per_image:.2f} ms/image\n")
            f.write(f"実行プロバイダー: {providers}\n")

        print(f"サマリーを保存: {summary_path}")


if __name__ == "__main__":
    main()
