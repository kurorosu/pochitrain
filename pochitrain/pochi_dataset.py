"""
pochitrain.pochi_dataset: Pochiデータセット.

カスタムデータセット用のシンプルなクラスと基本的なtransform
"""

import logging
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, decode_image

from pochitrain.logging import LoggerManager

logger: logging.Logger = LoggerManager().get_logger(__name__)

_PIL_ONLY_TRANSFORMS: Tuple[type, ...] = (transforms.ToPILImage,)


def _check_pil_transform(t: Any, dataset_name: str) -> bool:
    """PIL専用transformが含まれているかチェックし, 警告をログ出力する.

    Args:
        t: チェックするtransform
        dataset_name: フォールバック元となるデータセット名

    Returns:
        PIL専用transformであれば True, そうでなければ False
    """
    if isinstance(t, _PIL_ONLY_TRANSFORMS):
        action = (
            "PochiImageDatasetにフォールバックします."
            if dataset_name == "FastInferenceDataset"
            else "フォールバックします."
        )
        logger.warning(
            f"PIL専用transform {type(t).__name__} が含まれています. "
            f"{dataset_name}は使用できないため, {action}"
        )
        return True
    return False


class PochiImageDataset(Dataset):
    """
    Pochi画像分類データセット.

    フォルダ構造:
    root/
        class1/
            image1.jpg
            image2.jpg
        class2/
            image1.jpg
            image2.jpg

    Args:
        root (str): データのルートディレクトリ
        transform (callable, optional): データ変換関数
        extensions (tuple): 許可する拡張子
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
    ):
        """PochiImageDatasetを初期化."""
        self.root = Path(root)
        self.transform = transform
        self.extensions = extensions

        self.image_paths: List[Path] = []
        self.labels: List[int] = []
        self.classes: List[str] = []

        self._load_data()

    def _load_data(self) -> None:
        """データの読み込み."""
        if not self.root.exists():
            raise FileNotFoundError(f"データディレクトリが見つかりません: {self.root}")

        class_folders = [d for d in self.root.iterdir() if d.is_dir()]
        class_folders.sort()

        if not class_folders:
            raise ValueError(f"クラスフォルダが見つかりません: {self.root}")

        self.classes = [folder.name for folder in class_folders]
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        for class_folder in class_folders:
            class_idx = class_to_idx[class_folder.name]

            for image_path in class_folder.iterdir():
                if image_path.suffix.lower() in self.extensions:
                    self.image_paths.append(image_path)
                    self.labels.append(class_idx)

        if not self.image_paths:
            raise ValueError(f"画像ファイルが見つかりません: {self.root}")

    def __len__(self) -> int:
        """データセットのサイズを返す."""
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[Union[Tensor, Image.Image], int]:
        """指定されたインデックスのデータを返す."""
        image_path = self.image_paths[index]
        label = self.labels[index]

        image: Union[Tensor, Image.Image] = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_classes(self) -> List[str]:
        """クラス名のリストを取得."""
        return self.classes

    def get_class_counts(self) -> dict:
        """各クラスの画像数を取得."""
        counts = {}
        for class_name in self.classes:
            counts[class_name] = self.labels.count(self.classes.index(class_name))
        return counts

    def get_file_paths(self) -> List[str]:
        """
        データセット内のすべてのファイルパスを取得.

        Returns:
            List[str]: ファイルパスのリスト
        """
        return [str(path) for path in self.image_paths]


class FastInferenceDataset(PochiImageDataset):
    """推論CLI専用の高速画像データセット.

    torchvision.io.decode_image を使用してPIL経由のオーバーヘッドを排除する.
    decode_image は直接テンソルを返すため, ToTensor() は不要.
    代わりに ConvertImageDtype + Normalize で前処理を行う.

    Note:
        本クラスはテンソル入力に対応した transform を前提とする.
        PIL.Image を前提とする独自 transform を含む場合は実行時エラーとなる.
        その場合は PochiImageDataset の利用を推奨する.

    Args:
        root: データのルートディレクトリ
        transform: データ変換関数 (ConvertImageDtype + Normalize 等, ToTensor不要)
        extensions: 許可する拡張子
    """

    _transform_error_logged: bool = False

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        """指定されたインデックスのデータを返す."""
        image_path = self.image_paths[index]
        label = self.labels[index]

        image = decode_image(str(image_path), mode=ImageReadMode.RGB)

        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                cls = type(self)
                if not cls._transform_error_logged:
                    dataset_name = cls.__name__
                    logger.error(
                        f"{dataset_name} で transform 実行に失敗しました: {e}. "
                        "PIL前提の transform が含まれている可能性があります. "
                        "PochiImageDataset への切替を検討してください."
                    )
                    cls._transform_error_logged = True
                raise

        return image, label


class GpuInferenceDataset(FastInferenceDataset):
    """GPU推論専用の高速画像データセット.

    decode_image で画像をデコードし, uint8テンソルをそのまま返す.
    GPU上での正規化は gpu_normalize() で別途実行する.
    transform は適用しない (Resize等が必要な場合のみ指定可能).

    Args:
        root: データのルートディレクトリ
        transform: リサイズ等のCPU上テンソル変換 (Normalize, ConvertImageDtype は不要)
        extensions: 許可する拡張子
    """

    pass


def extract_normalize_params(
    transform: Callable[..., Any],
) -> Tuple[List[float], List[float]]:
    """Compose内のNormalizeからmean, stdを抽出する.

    Args:
        transform: configのval_transform (Compose or 単体transform)

    Returns:
        (mean, std) のタプル

    Raises:
        ValueError: Normalizeが見つからない場合
    """
    if hasattr(transform, "transforms"):
        for t in transform.transforms:
            if isinstance(t, transforms.Normalize):
                return list(t.mean), list(t.std)
    elif isinstance(transform, transforms.Normalize):
        return list(transform.mean), list(transform.std)
    raise ValueError("transformにNormalizeが含まれていません")


def build_gpu_preprocess_transform(
    transform: Callable[..., Any],
) -> Optional[transforms.Compose]:
    """configのval_transformからGpuInferenceDataset向けのtransformを構築.

    Normalize, ToTensor, ConvertImageDtype を除外し,
    Resize / CenterCrop 等のテンソル互換transformのみを残す.
    PIL専用transformが含まれる場合はNoneを返す.

    Args:
        transform: configのval_transform

    Returns:
        GpuInferenceDataset向けのtransform. PIL専用transform時はNone.
        残すtransformが無い場合は空のCompose (transforms=[]) を返す.
    """
    _SKIP_TRANSFORMS = (
        transforms.ToTensor,
        transforms.Normalize,
        transforms.ConvertImageDtype,
    )

    new_transforms: List[Any] = []
    if hasattr(transform, "transforms"):
        for t in transform.transforms:
            if _check_pil_transform(t, "GpuInferenceDataset"):
                return None
            elif isinstance(t, _SKIP_TRANSFORMS):
                continue
            else:
                new_transforms.append(t)
    else:
        if _check_pil_transform(transform, "GpuInferenceDataset"):
            return None
        elif not isinstance(transform, _SKIP_TRANSFORMS):
            new_transforms.append(transform)

    return transforms.Compose(new_transforms)


def gpu_normalize(
    images: Tensor,
    mean_255: Tensor,
    std_255: Tensor,
    non_blocking: bool = True,
) -> Tensor:
    """uint8テンソルをfloat変換+正規化する.

    decode_image が返す uint8 テンソルを受け取り,
    float32 変換と正規化を一括で行う.
    `mean_255`, `std_255` は事前計算済みテンソルを受け取り,
    ループ内での再生成を回避する.

    Args:
        images: uint8テンソル (C,H,W) or (N,C,H,W)
        mean_255: 255倍済み平均テンソル (1,3,1,1)
        std_255: 255倍済み標準偏差テンソル (1,3,1,1)
        non_blocking: 変換時に non_blocking 転送を使うかどうか.

    Returns:
        正規化済み float32 テンソル (N,C,H,W)
    """
    if images.dim() == 3:
        images = images.unsqueeze(0)

    device = mean_255.device
    images_float = images.to(
        device=device,
        dtype=torch.float32,
        non_blocking=non_blocking,
    )
    return (images_float - mean_255) / std_255


def create_scaled_normalize_tensors(
    mean: List[float],
    std: List[float],
    device: Union[str, torch.device] = "cuda",
) -> Tuple[Tensor, Tensor]:
    """正規化用の 255 倍済みテンソルを作成する.

    Args:
        mean: 正規化の平均値 (例: [0.485, 0.456, 0.406])
        std: 正規化の標準偏差 (例: [0.229, 0.224, 0.225])
        device: テンソル配置先デバイス

    Returns:
        (mean_255, std_255) のタプル. 形状はどちらも (1,3,1,1)
    """
    mean_255 = torch.tensor(mean, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std_255 = torch.tensor(std, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    return mean_255 * 255.0, std_255 * 255.0


def convert_transform_for_fast_inference(
    transform: Callable[..., Any],
) -> Optional[transforms.Compose]:
    """configのval_transformをFastInferenceDataset向けに変換.

    ToTensor() を ConvertImageDtype(float32) に置き換える.
    decode_image は直接テンソルを返すため, ToTensor() は不要.

    PIL専用のtransformが含まれている場合はNoneを返し,
    呼び出し側でPochiImageDatasetへフォールバックさせる.

    Composeでない単体transformの場合は, そのtransformを保持しつつ
    ConvertImageDtype(float32) を末尾に追加する.

    Args:
        transform: configのval_transform (Compose or 単体transform)

    Returns:
        FastInferenceDataset向けのtransform. PIL専用transformが含まれる場合はNone.
    """
    new_transforms: List[Any] = []
    if hasattr(transform, "transforms"):
        for t in transform.transforms:
            if isinstance(t, transforms.ToTensor):
                new_transforms.append(transforms.ConvertImageDtype(torch.float32))
            elif _check_pil_transform(t, "FastInferenceDataset"):
                return None
            else:
                new_transforms.append(t)
    else:
        if isinstance(transform, transforms.ToTensor):
            new_transforms.append(transforms.ConvertImageDtype(torch.float32))
        elif _check_pil_transform(transform, "FastInferenceDataset"):
            return None
        else:
            new_transforms.append(transform)
            new_transforms.append(transforms.ConvertImageDtype(torch.float32))
    return transforms.Compose(new_transforms)


def get_basic_transforms(
    image_size: int = 224,
    is_training: bool = True,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
) -> transforms.Compose:
    """
    基本的なデータ変換を取得.

    Args:
        image_size (int): 画像サイズ
        is_training (bool): 訓練用かどうか
        mean (List[float]): 正規化の平均値
        std (List[float]): 正規化の標準偏差

    Returns:
        transforms.Compose: 変換処理
    """
    if is_training:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.14)),  # 256 for 224
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )


def create_data_loaders(
    train_root: str,
    val_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    train_transform: Optional[Callable[..., Any]] = None,
    val_transform: Optional[Callable[..., Any]] = None,
) -> Tuple[DataLoader[Any], DataLoader[Any], List[str]]:
    """
    データローダーを作成.

    Args:
        train_root (str): 訓練データのルートディレクトリ
        val_root (str): 検証データのルートディレクトリ（必須）
        batch_size (int): バッチサイズ
        num_workers (int): ワーカー数
        pin_memory (bool): メモリピニング
        train_transform (transforms.Compose): 訓練用変換（必須）
        val_transform (transforms.Compose): 検証用変換（必須）

    Returns:
        Tuple[DataLoader, DataLoader, List[str]]: (訓練ローダー, 検証ローダー, クラス名)
    """
    train_dataset = PochiImageDataset(train_root, transform=train_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_dataset = PochiImageDataset(val_root, transform=val_transform)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, train_dataset.get_classes()
