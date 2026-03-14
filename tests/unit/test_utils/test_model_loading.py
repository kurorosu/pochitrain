"""model_loading ユーティリティのテスト."""

from unittest.mock import patch

import pytest
import torch

from pochitrain.utils.model_loading import (
    _load_torch_checkpoint,
    load_model_from_checkpoint,
)


class TestLoadTorchCheckpoint:
    """_load_torch_checkpoint のテストクラス."""

    def test_weights_only_supported(self, tmp_path):
        """weights_only=True で正常に読み込めるケース."""
        checkpoint_path = tmp_path / "model.pth"
        data = {"key": "value"}
        torch.save(data, checkpoint_path)

        result = _load_torch_checkpoint(checkpoint_path, torch.device("cpu"))

        assert result == data

    def test_type_error_fallback(self, tmp_path):
        """weights_only 未対応時にフォールバックして正しく読み込めるケース."""
        checkpoint_path = tmp_path / "model.pth"
        expected = {"key": "value"}
        torch.save(expected, checkpoint_path)

        original_load = torch.load

        def load_without_weights_only(*args, **kwargs):
            if "weights_only" in kwargs:
                raise TypeError("unexpected keyword argument 'weights_only'")
            return original_load(*args, **kwargs)

        with patch(
            "pochitrain.utils.model_loading.torch.load",
            side_effect=load_without_weights_only,
        ):
            result = _load_torch_checkpoint(checkpoint_path, torch.device("cpu"))

        assert result == expected


class TestLoadModelFromCheckpoint:
    """load_model_from_checkpoint のテストクラス."""

    @pytest.fixture
    def model(self):
        """テスト用のシンプルなモデル."""
        return torch.nn.Linear(10, 2)

    def test_load_dict_checkpoint(self, tmp_path, model):
        """dict形式のチェックポイントからモデルを復元するケース."""
        checkpoint_path = tmp_path / "model.pth"
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": 5,
            "best_accuracy": 0.95,
        }
        torch.save(checkpoint, checkpoint_path)

        new_model = torch.nn.Linear(10, 2)
        metadata = load_model_from_checkpoint(
            new_model, checkpoint_path, torch.device("cpu")
        )

        assert metadata["epoch"] == 5
        assert metadata["best_accuracy"] == 0.95
        # state_dict が正しく適用されたことを確認
        for key in model.state_dict():
            assert torch.equal(new_model.state_dict()[key], model.state_dict()[key])

    def test_load_dict_checkpoint_without_metadata(self, tmp_path, model):
        """メタデータなしのdict形式チェックポイント."""
        checkpoint_path = tmp_path / "model.pth"
        checkpoint = {"model_state_dict": model.state_dict()}
        torch.save(checkpoint, checkpoint_path)

        new_model = torch.nn.Linear(10, 2)
        metadata = load_model_from_checkpoint(
            new_model, checkpoint_path, torch.device("cpu")
        )

        assert metadata == {}

    def test_load_raw_state_dict(self, tmp_path, model):
        """state_dict 直接保存形式からモデルを復元するケース."""
        checkpoint_path = tmp_path / "model.pth"
        torch.save(model.state_dict(), checkpoint_path)

        new_model = torch.nn.Linear(10, 2)
        metadata = load_model_from_checkpoint(
            new_model, checkpoint_path, torch.device("cpu")
        )

        assert metadata == {}
        for key in model.state_dict():
            assert torch.equal(new_model.state_dict()[key], model.state_dict()[key])

    def test_file_not_found(self, tmp_path, model):
        """存在しないチェックポイントで FileNotFoundError."""
        checkpoint_path = tmp_path / "nonexistent.pth"

        with pytest.raises(FileNotFoundError, match="モデルファイルが見つかりません"):
            load_model_from_checkpoint(model, checkpoint_path, torch.device("cpu"))
