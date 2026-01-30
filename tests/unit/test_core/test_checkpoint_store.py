"""CheckpointStoreクラスのユニットテスト."""

import logging
import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn, optim

from pochitrain.training.checkpoint_store import CheckpointStore


@pytest.fixture
def work_dir(tmp_path: Path) -> Path:
    """テスト用の作業ディレクトリ."""
    return tmp_path


@pytest.fixture
def logger() -> logging.Logger:
    """テスト用ロガー."""
    return logging.getLogger("test_checkpoint_store")


@pytest.fixture
def store(work_dir: Path, logger: logging.Logger) -> CheckpointStore:
    """CheckpointStoreインスタンス."""
    return CheckpointStore(work_dir, logger)


@pytest.fixture
def model() -> nn.Module:
    """テスト用の簡易モデル."""
    return nn.Linear(10, 2)


@pytest.fixture
def optimizer(model: nn.Module) -> optim.Optimizer:
    """テスト用オプティマイザ."""
    return optim.SGD(model.parameters(), lr=0.01)


class TestSaveCheckpoint:
    """save_checkpointメソッドのテスト."""

    def test_save_checkpoint(
        self,
        store: CheckpointStore,
        model: nn.Module,
        optimizer: optim.Optimizer,
        work_dir: Path,
    ) -> None:
        """チェックポイントファイルが正しく保存される."""
        path = store.save_checkpoint(
            filename="checkpoint.pth",
            epoch=5,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            best_accuracy=85.0,
        )

        assert path == work_dir / "checkpoint.pth"
        assert path.exists()

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        assert checkpoint["epoch"] == 5
        assert checkpoint["best_accuracy"] == 85.0
        assert checkpoint["model_state_dict"] is not None
        assert checkpoint["optimizer_state_dict"] is not None

    def test_save_checkpoint_with_scheduler(
        self,
        store: CheckpointStore,
        model: nn.Module,
        optimizer: optim.Optimizer,
        work_dir: Path,
    ) -> None:
        """スケジューラー状態も保存される."""
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5)

        path = store.save_checkpoint(
            filename="checkpoint.pth",
            epoch=3,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_accuracy=90.0,
        )

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        assert "scheduler_state_dict" in checkpoint
        assert checkpoint["scheduler_state_dict"] is not None

    def test_save_checkpoint_without_scheduler(
        self,
        store: CheckpointStore,
        model: nn.Module,
        optimizer: optim.Optimizer,
        work_dir: Path,
    ) -> None:
        """スケジューラーなしの場合, scheduler_state_dictが含まれない."""
        path = store.save_checkpoint(
            filename="checkpoint.pth",
            epoch=1,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            best_accuracy=50.0,
        )

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        assert "scheduler_state_dict" not in checkpoint


class TestSaveBestModel:
    """save_best_modelメソッドのテスト."""

    def test_save_best_model(
        self,
        store: CheckpointStore,
        model: nn.Module,
        optimizer: optim.Optimizer,
        work_dir: Path,
    ) -> None:
        """ベストモデルが保存される."""
        store.save_best_model(
            epoch=10,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            best_accuracy=95.0,
        )

        best_file = work_dir / "best_epoch10.pth"
        assert best_file.exists()

    def test_save_best_model_deletes_existing(
        self,
        store: CheckpointStore,
        model: nn.Module,
        optimizer: optim.Optimizer,
        work_dir: Path,
    ) -> None:
        """既存のベストモデルファイルが削除される."""
        # 最初のベストモデルを保存
        store.save_best_model(
            epoch=5,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            best_accuracy=80.0,
        )
        assert (work_dir / "best_epoch5.pth").exists()

        # 新しいベストモデルを保存
        store.save_best_model(
            epoch=10,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            best_accuracy=90.0,
        )

        # 古いファイルが削除され, 新しいファイルが存在する
        assert not (work_dir / "best_epoch5.pth").exists()
        assert (work_dir / "best_epoch10.pth").exists()


class TestSaveLastModel:
    """save_last_modelメソッドのテスト."""

    def test_save_last_model(
        self,
        store: CheckpointStore,
        model: nn.Module,
        optimizer: optim.Optimizer,
        work_dir: Path,
    ) -> None:
        """ラストモデルが上書き保存される."""
        store.save_last_model(
            epoch=1,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            best_accuracy=60.0,
        )

        last_file = work_dir / "last_model.pth"
        assert last_file.exists()

        # 上書き保存
        store.save_last_model(
            epoch=5,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            best_accuracy=80.0,
        )

        assert last_file.exists()
        checkpoint = torch.load(last_file, map_location="cpu", weights_only=True)
        assert checkpoint["epoch"] == 5
        assert checkpoint["best_accuracy"] == 80.0


class TestLoadCheckpoint:
    """load_checkpointメソッドのテスト."""

    def test_load_checkpoint(
        self,
        store: CheckpointStore,
        model: nn.Module,
        optimizer: optim.Optimizer,
        work_dir: Path,
    ) -> None:
        """チェックポイントが正しく復元される."""
        # 保存
        store.save_checkpoint(
            filename="checkpoint.pth",
            epoch=7,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            best_accuracy=88.5,
        )

        # 新しいモデルにロード
        new_model = nn.Linear(10, 2)
        new_optimizer = optim.SGD(new_model.parameters(), lr=0.01)

        result = store.load_checkpoint(
            filename="checkpoint.pth",
            model=new_model,
            device=torch.device("cpu"),
            optimizer=new_optimizer,
        )

        assert result["epoch"] == 7
        assert result["best_accuracy"] == 88.5

        # モデルの重みが復元されていることを確認
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.equal(p1, p2)

    def test_load_checkpoint_with_scheduler(
        self,
        store: CheckpointStore,
        model: nn.Module,
        optimizer: optim.Optimizer,
        work_dir: Path,
    ) -> None:
        """スケジューラー付きチェックポイントが正しく復元される."""
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5)
        # 数ステップ進める
        scheduler.step()
        scheduler.step()

        store.save_checkpoint(
            filename="checkpoint.pth",
            epoch=2,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_accuracy=70.0,
        )

        # 新しいモデル・スケジューラーにロード
        new_model = nn.Linear(10, 2)
        new_optimizer = optim.SGD(new_model.parameters(), lr=0.01)
        new_scheduler = optim.lr_scheduler.StepLR(new_optimizer, step_size=5)

        store.load_checkpoint(
            filename="checkpoint.pth",
            model=new_model,
            device=torch.device("cpu"),
            optimizer=new_optimizer,
            scheduler=new_scheduler,
        )

        # スケジューラーの状態が復元されていることを確認
        assert new_scheduler.state_dict() == scheduler.state_dict()

    def test_load_checkpoint_file_not_found(
        self,
        store: CheckpointStore,
        model: nn.Module,
    ) -> None:
        """存在しないファイルでFileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="チェックポイントが見つかりません"):
            store.load_checkpoint(
                filename="nonexistent.pth",
                model=model,
                device=torch.device("cpu"),
            )
