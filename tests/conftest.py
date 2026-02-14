"""テスト共通フィクスチャ."""

from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def create_dummy_dataset(tmp_path: Path):
    """ダミー画像データセットを作成するファクトリフィクスチャ.

    Args:
        tmp_path: pytest組み込みの一時ディレクトリ.

    Returns:
        データセット作成関数. 引数:
            structure: クラス名→画像数の辞書 (例: {"cat": 3, "dog": 2})
            image_size: 画像サイズのタプル (デフォルト: (32, 32))
            subdir: サブディレクトリ名 (例: "train"). Noneならtmp_path直下に作成.

    Example:
        >>> def test_example(create_dummy_dataset):
        ...     path = create_dummy_dataset({"cat": 3, "dog": 2})
        ...     assert (path / "cat").exists()
    """

    def _create(
        structure: dict[str, int],
        *,
        image_size: tuple[int, int] = (32, 32),
        subdir: str | None = None,
    ) -> Path:
        base = tmp_path / subdir if subdir else tmp_path
        for class_name, num_images in structure.items():
            class_dir = base / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for i in range(num_images):
                img = Image.new("RGB", image_size, color=(i * 50 % 255, 100, 150))
                img.save(class_dir / f"image_{i}.jpg")
        return base

    return _create


@pytest.fixture
def create_dummy_train_val(create_dummy_dataset):
    """train/val構造のダミーデータセットを作成するファクトリフィクスチャ.

    Args:
        create_dummy_dataset: ベースとなるデータセット作成フィクスチャ.

    Returns:
        データセット作成関数. 引数:
            classes: クラス名のリスト (デフォルト: ["cat", "dog"])
            train_per_class: trainの1クラスあたりの画像数 (デフォルト: 3)
            val_per_class: valの1クラスあたりの画像数 (デフォルト: 2)
            image_size: 画像サイズのタプル (デフォルト: (64, 64))

    Example:
        >>> def test_example(create_dummy_train_val):
        ...     train_root, val_root = create_dummy_train_val()
    """

    def _create(
        classes: list[str] | None = None,
        *,
        train_per_class: int = 3,
        val_per_class: int = 2,
        image_size: tuple[int, int] = (64, 64),
    ) -> tuple[str, str]:
        if classes is None:
            classes = ["cat", "dog"]

        train_structure = {c: train_per_class for c in classes}
        val_structure = {c: val_per_class for c in classes}

        train_path = create_dummy_dataset(
            train_structure, image_size=image_size, subdir="train"
        )
        val_path = create_dummy_dataset(
            val_structure, image_size=image_size, subdir="val"
        )
        return str(train_path), str(val_path)

    return _create
