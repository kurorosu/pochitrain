#!/usr/bin/env python3
"""
PyTorchモデル(.pth)をONNX形式に変換するスクリプト.

使用例:
    python tools/export_onnx.py work_dirs/20251018_001/models/best_epoch40.pth
    python tools/export_onnx.py model.pth --config work_dirs/20251018_001/config.py
    python tools/export_onnx.py model.pth --model-name resnet18 --num-classes 4
    python tools/export_onnx.py model.pth --input-size 256 256
"""

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import torch


def load_config(config_path: str) -> Dict[str, Any]:
    """設定ファイルを読み込む."""
    config_path_obj = Path(config_path)

    if not config_path_obj.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")

    spec = importlib.util.spec_from_file_location("config", config_path_obj)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"設定ファイルの読み込みに失敗しました: {config_path}")

    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    config = {}
    for key in dir(config_module):
        if not key.startswith("_"):
            value = getattr(config_module, key)
            if not callable(value) or hasattr(value, "transforms"):
                config[key] = value

    return config


def export_to_onnx(
    model: torch.nn.Module,
    output_path: Path,
    input_size: Tuple[int, int],
    device: torch.device,
    opset_version: int = 17,
) -> Path:
    """モデルをONNX形式でエクスポート."""
    model.eval()

    # ダミー入力を作成 (batch=1, channels=3, height, width)
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1], device=device)

    # ONNX形式でエクスポート (TorchScriptベースの従来エクスポーター)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        dynamo=False,  # TorchScriptベースのエクスポーターを使用
    )

    return output_path


def verify_onnx_model(
    onnx_path: Path,
    pytorch_model: torch.nn.Module,
    input_size: Tuple[int, int],
    device: torch.device,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> bool:
    """エクスポートしたONNXモデルを検証.

    Args:
        onnx_path: ONNXモデルのパス
        pytorch_model: 元のPyTorchモデル
        input_size: 入力サイズ (height, width)
        device: PyTorchモデルのデバイス
        rtol: 相対許容誤差
        atol: 絶対許容誤差

    Returns:
        検証成功の場合True
    """
    # 1. ONNXモデルの構造検証
    print("ONNXモデルの構造を検証中...")
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("構造検証: OK")

    # 2. PyTorchとONNXの出力比較
    print("PyTorchとONNXの出力を比較中...")
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1], device=device)

    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(dummy_input).cpu().numpy()

    # ONNXセッション作成
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_output = session.run(None, {"input": dummy_input.cpu().numpy()})[0]

    # 出力の比較
    is_close: bool = bool(
        np.allclose(pytorch_output, onnx_output, rtol=rtol, atol=atol)
    )

    if is_close:
        print("出力比較: OK")
        max_diff = np.max(np.abs(pytorch_output - onnx_output))
        print(f"最大差分: {max_diff:.2e}")
    else:
        print("出力比較: NG")
        max_diff = np.max(np.abs(pytorch_output - onnx_output))
        mean_diff = np.mean(np.abs(pytorch_output - onnx_output))
        print(f"最大差分: {max_diff:.2e}, 平均差分: {mean_diff:.2e}")

    return is_close


def main() -> None:
    """メイン関数."""
    parser = argparse.ArgumentParser(
        description="PyTorchモデル(.pth)をONNX形式に変換",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 設定ファイルから情報を取得（推奨）
  python tools/export_onnx.py work_dirs/20251018_001/models/best_epoch40.pth

  # 設定ファイルを明示的に指定
  python tools/export_onnx.py model.pth --config work_dirs/20251018_001/config.py

  # モデル情報を直接指定
  python tools/export_onnx.py model.pth --model-name resnet18 --num-classes 4

  # 入力サイズを指定
  python tools/export_onnx.py model.pth --input-size 256 256

  # 出力先を指定
  python tools/export_onnx.py model.pth -o output/model.onnx
        """,
    )

    parser.add_argument("model_path", help="変換するPyTorchモデルファイル(.pth)")
    parser.add_argument(
        "--config",
        "-c",
        help="設定ファイルパス（省略時はモデルと同階層のconfig.pyを探索）",
    )
    parser.add_argument(
        "--model-name",
        default="resnet18",
        help="モデル名 (default: resnet18)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        help="分類クラス数（設定ファイルから取得できない場合に必要）",
    )
    parser.add_argument(
        "--input-size",
        nargs=2,
        type=int,
        required=True,
        metavar=("HEIGHT", "WIDTH"),
        help="入力画像サイズ（必須）",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="出力ファイルパス (default: 入力ファイルと同じ場所に.onnx拡張子で保存)",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=17,
        help="ONNXオペセットバージョン (default: 17)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="使用デバイス (default: cpu)",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="エクスポート後の検証をスキップ",
    )

    args = parser.parse_args()

    # モデルパスの確認
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"エラー: モデルファイルが見つかりません: {model_path}")
        sys.exit(1)

    # 設定ファイルの探索
    config = None
    if args.config:
        config_path = Path(args.config)
    else:
        # モデルと同階層または親ディレクトリからconfig.pyを探索
        possible_paths = [
            model_path.parent / "config.py",
            model_path.parent.parent / "config.py",
        ]
        config_path = None
        for p in possible_paths:
            if p.exists():
                config_path = p
                break

    if config_path and config_path.exists():
        try:
            config = load_config(str(config_path))
            print(f"設定ファイルを読み込み: {config_path}")
        except Exception as e:
            print(f"警告: 設定ファイルの読み込みに失敗: {e}")
            config = None

    # モデル情報の取得
    model_name = args.model_name
    num_classes = args.num_classes

    if config:
        model_name = config.get("model_name", model_name)
        if num_classes is None:
            num_classes = config.get("num_classes")

    if num_classes is None:
        print("エラー: --num-classes を指定するか、設定ファイルを使用してください")
        sys.exit(1)

    # 入力サイズの決定
    if args.input_size:
        input_size = (args.input_size[0], args.input_size[1])
    else:
        # デフォルトは224x224
        input_size = (224, 224)

    # 出力パスの決定
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = model_path.with_suffix(".onnx")

    # デバイス設定
    device = torch.device(args.device)

    print(f"モデル: {model_name}")
    print(f"クラス数: {num_classes}")
    print(f"入力サイズ: {input_size[0]}x{input_size[1]}")
    print(f"デバイス: {device}")
    print(f"出力先: {output_path}")

    # モデルの作成と重みの読み込み
    try:
        # pochitrainのモデル作成関数を使用
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from pochitrain.models.pochi_models import create_model

        model = create_model(model_name, num_classes, pretrained=False)
        model.to(device)

        # チェックポイントの読み込み
        checkpoint = torch.load(model_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            if "best_accuracy" in checkpoint:
                print(f"モデル精度: {checkpoint['best_accuracy']:.2f}%")
            if "epoch" in checkpoint:
                print(f"エポック: {checkpoint['epoch']}")
        else:
            # state_dictが直接保存されている場合
            model.load_state_dict(checkpoint)

        print("モデルの読み込み完了")

    except Exception as e:
        print(f"エラー: モデルの読み込みに失敗: {e}")
        sys.exit(1)

    # ONNX変換
    try:
        print("ONNX変換を実行中...")
        export_to_onnx(
            model=model,
            output_path=output_path,
            input_size=input_size,
            device=device,
            opset_version=args.opset_version,
        )
        print(f"ONNX変換完了: {output_path}")

        # ファイルサイズを表示
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"ファイルサイズ: {file_size_mb:.2f} MB")

    except Exception as e:
        print(f"エラー: ONNX変換に失敗: {e}")
        sys.exit(1)

    # ONNX検証
    if not args.skip_verify:
        print("\n--- ONNX検証 ---")
        try:
            is_valid = verify_onnx_model(
                onnx_path=output_path,
                pytorch_model=model,
                input_size=input_size,
                device=device,
            )
            if is_valid:
                print("検証完了: ONNXモデルは正常です")
            else:
                print("警告: PyTorchとONNXの出力に差異があります")
                sys.exit(1)
        except Exception as e:
            print(f"エラー: ONNX検証に失敗: {e}")
            sys.exit(1)
    else:
        print("\n検証をスキップしました")


if __name__ == "__main__":
    main()
