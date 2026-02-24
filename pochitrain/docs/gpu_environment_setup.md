# GPU 環境セットアップガイド

pochitrain で GPU 訓練・ONNX Runtime GPU 推論・TensorRT 推論を利用するために必要な外部依存のインストール手順.

## 検証済み環境

| 構成 | PyTorch | CUDA Toolkit | cuDNN | TensorRT | OS |
|------|---------|-------------|-------|----------|----|
| A | 2.9.1 | 12.9 | 9.19 | 10.14.1 | Windows 11 |
| B | 2.9.1 | 12.1 | 8.8.1 | 10.14.1 | Windows 11 |
| C | 2.5 | 12.6 | 9.3 | 10.3 | Linux (Jetson JetPack 6.2.1) |

> **Note**: PyTorch・CUDA・cuDNN・TensorRT はバージョン間の互換性が厳密.
> 上記の検証済み組み合わせを推奨する.

## 1. CUDA Toolkit

### 概要

PyTorch の GPU 演算および TensorRT のビルドに必要.

### インストール

1. [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) からダウンロード
2. インストーラーを実行 (Windows: `.exe`, Linux: `.run` または パッケージマネージャー)

> インストーラーが `PATH` (`nvcc` 等) や `CUDA_PATH` を自動設定するため, 手動での環境変数設定は不要の可能性あり.
> インストール後は以下のコマンドで環境変数が正しく設定されているか確認すること.

### 動作確認

```bash
nvcc --version
# 例: Cuda compilation tools, release 12.9, V12.9.xxx
```

環境変数を確認:

```bash
# Windows (PowerShell)
echo $env:CUDA_PATH
# 例: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9

echo $env:Path -split ";" | Select-String "CUDA"
# CUDA 関連のパスが含まれていること

# Linux
echo $CUDA_HOME
echo $PATH | tr ":" "\n" | grep cuda
```

## 2. cuDNN

### 概要

PyTorch の畳み込み演算を高速化する NVIDIA ライブラリ. CUDA Toolkit とバージョンの互換性がある.

### インストール

1. [NVIDIA cuDNN Archive](https://developer.nvidia.com/cudnn-archive) からダウンロード (NVIDIA アカウント要)
2. インストーラーを実行、または ZIP/tar を展開し CUDA Toolkit のインストールディレクトリにファイルをコピー

> インストーラーを実行した場合は, 手動での環境変数設定は不要の可能性あり.
> ZIP/tar の場合は CUDA ディレクトリにファイルをコピーするだけで完了.
> いずれの場合も, インストール後は以下のコマンドで正しく認識されているか確認すること.

### 動作確認

```python
import torch
print(torch.backends.cudnn.version())  # 例: 90900
print(torch.backends.cudnn.is_available())  # True
```

## 3. TensorRT SDK

### 概要

ONNX モデルを最適化し高速推論を行う NVIDIA のランタイム. pochitrain では `pochi convert` および `infer-trt` コマンドで利用する.

> TensorRT を使わず PyTorch / ONNX Runtime のみで運用する場合は, このセクションはスキップ可能.

### インストール

1. [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) から ZIP をダウンロード
2. 任意のディレクトリに展開 (例: `D:\NVIDIA\TensorRT-10.14.1.48`)

### 環境変数の設定 (手動で必須)

TensorRT は ZIP 配布のため, インストーラーによる自動設定は行われない. 以下の環境変数を手動で設定する.

#### Windows

システム環境変数の `Path` に以下を追加:

```
<TensorRT展開先>\lib
```

例:
```
D:\NVIDIA\TensorRT-10.14.1.48\lib
```

#### Linux

`~/.bashrc` または `~/.profile` に以下を追記:

```bash
export LD_LIBRARY_PATH=<TensorRT展開先>/lib:$LD_LIBRARY_PATH
export PATH=<TensorRT展開先>/bin:$PATH
```

例:
```bash
export LD_LIBRARY_PATH=/opt/TensorRT-10.14.1.48/lib:$LD_LIBRARY_PATH
export PATH=/opt/TensorRT-10.14.1.48/bin:$PATH
```

設定を反映:
```bash
source ~/.bashrc
```

### Python API のインストール

```bash
uv pip install <TensorRT展開先>/python/tensorrt-<version>-cpXX-none-<platform>.whl
```

例 (Windows, Python 3.13):
```bash
uv pip install D:/NVIDIA/TensorRT-10.14.1.48/python/tensorrt-10.14.1.48-cp313-none-win_amd64.whl
```

> **Warning**: `uv sync` を実行すると TensorRT がアンインストールされる.
> `uv sync` 後は必ず上記コマンドで再インストールすること.

### 動作確認

```bash
# trtexec が使えることを確認
trtexec --help

# Python API の確認
python -c "import tensorrt; print(tensorrt.__version__)"
```

## 4. ONNX Runtime (GPU)

### 概要

ONNX モデルの GPU 推論に必要なランタイム. pochitrain では `infer-onnx` コマンドで利用する.

> ONNX Runtime を使わず PyTorch / TensorRT のみで運用する場合は, このセクションはスキップ可能.

### インストール

#### Windows

```bash
pip install onnxruntime-gpu
```

#### Jetson (JetPack 6.2.1)

PyPI の標準パッケージは x86_64 向けのため, NVIDIA の Jetson 向けインデックスを指定する.

```bash
pip install onnxruntime-gpu --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```

> **Note**: JetPack バージョンによってインデックス URL が異なる場合がある.
> 最新の URL は [Jetson AI Lab](https://pypi.jetson-ai-lab.io/) で確認すること.

### 動作確認

```python
import onnxruntime as ort
print(ort.__version__)
print(ort.get_available_providers())
# CUDAExecutionProvider が含まれていれば GPU 推論が可能
```

## 5. Jetson ベンチマーク時の電力・クロック固定

Jetson で推論ベンチマークを行う場合は, `nvpmodel` と `jetson_clocks` を事前に適用する.
未適用だと動的電圧・周波数制御 (DVFS) により計測値が揺れ, `pure inference` の比較が不安定になる.

### 手順 (Jetson Orin Nano, JetPack 6.2.1)

```bash
# 現在のモード確認
sudo nvpmodel -q
sudo nvpmodel -q --verbose

# MAXN_SUPER (mode 2) へ設定
sudo nvpmodel -m 2

# クロック固定
sudo jetson_clocks

# 反映確認
sudo jetson_clocks --show
```

### オプション

- 温度影響を減らしたい場合: `sudo jetson_clocks --fan`
- 計測再現性を優先する場合: ベンチ実行前に毎回 `nvpmodel` と `jetson_clocks` を再適用する

### 注意

- `jetson_clocks` は再起動で解除される.
- `nvpmodel` は設定が残るため, 必要なら `sudo nvpmodel -m <元のmode_id>` で戻す.

## バージョン互換性

各コンポーネントのバージョンは相互に制約がある. 以下の公式ドキュメントで対応表を確認すること.

| コンポーネント | 互換性の確認先 |
|--------------|--------------|
| PyTorch ↔ CUDA | [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/) |
| CUDA ↔ cuDNN | [cuDNN Support Matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html) |
| TensorRT ↔ CUDA / cuDNN | [TensorRT Support Matrix](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html) |

## トラブルシューティング

### `import tensorrt` で `DLL load failed` / `cannot open shared object file`

TensorRT の共有ライブラリにパスが通っていない.

- **Windows**: システム環境変数 `Path` に `<TensorRT展開先>\lib` が含まれているか確認
- **Linux**: `LD_LIBRARY_PATH` に `<TensorRT展開先>/lib` が含まれているか確認

### `uv sync` 後に TensorRT が見つからない

`uv sync` は手動インストールしたパッケージを削除する. 再インストールが必要:

```bash
uv pip install <TensorRT展開先>/python/tensorrt-<version>-cpXX-none-<platform>.whl
```

### CUDA のバージョン不一致

PyTorch が期待する CUDA バージョンと, システムにインストールされた CUDA Toolkit のバージョンが異なる場合にエラーが発生する.

```bash
# PyTorch が使う CUDA バージョンを確認
python -c "import torch; print(torch.version.cuda)"

# システムの CUDA バージョンを確認
nvcc --version
```

両者のメジャー・マイナーバージョンが一致していることを確認すること.
