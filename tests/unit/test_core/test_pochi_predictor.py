"""PochiPredictorクラスのテスト.

実際のモデルとチェックポイントを使用した古典的テスト.
"""

from pathlib import Path

import pytest
import torch

from pochitrain.models.pochi_models import PochiModel
from pochitrain.pochi_predictor import PochiPredictor
from pochitrain.pochi_trainer import PochiTrainer


def _create_test_checkpoint(
    tmp_path: Path, model_name: str = "resnet18", num_classes: int = 3
) -> Path:
    """テスト用チェックポイントを作成.

    Args:
        tmp_path: 一時ディレクトリ
        model_name: モデル名
        num_classes: クラス数

    Returns:
        チェックポイントファイルのパス
    """
    model = PochiModel(model_name, num_classes, pretrained=False)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": 5,
        "best_accuracy": 85.0,
    }
    checkpoint_path = tmp_path / "models" / "best_epoch5.pth"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


class TestPochiPredictorInit:
    """PochiPredictor初期化のテスト."""

    def test_basic_init(self, tmp_path):
        """基本的な初期化が成功する."""
        checkpoint_path = _create_test_checkpoint(tmp_path)
        predictor = PochiPredictor(
            model_name="resnet18",
            num_classes=3,
            device="cpu",
            model_path=str(checkpoint_path),
        )
        assert predictor.model is not None
        assert predictor.model_path == checkpoint_path

    def test_model_in_eval_mode(self, tmp_path):
        """モデルがevalモードになっている."""
        checkpoint_path = _create_test_checkpoint(tmp_path)
        predictor = PochiPredictor(
            model_name="resnet18",
            num_classes=3,
            device="cpu",
            model_path=str(checkpoint_path),
        )
        assert not predictor.model.training

    def test_loads_best_accuracy(self, tmp_path):
        """best_accuracyがチェックポイントから復元される."""
        checkpoint_path = _create_test_checkpoint(tmp_path)
        predictor = PochiPredictor(
            model_name="resnet18",
            num_classes=3,
            device="cpu",
            model_path=str(checkpoint_path),
        )
        assert predictor.best_accuracy == 85.0

    def test_loads_epoch(self, tmp_path):
        """epochがチェックポイントから復元される."""
        checkpoint_path = _create_test_checkpoint(tmp_path)
        predictor = PochiPredictor(
            model_name="resnet18",
            num_classes=3,
            device="cpu",
            model_path=str(checkpoint_path),
        )
        assert predictor.epoch == 5

    def test_nonexistent_model_raises(self, tmp_path):
        """存在しないモデルパスでFileNotFoundErrorが発生する."""
        with pytest.raises(FileNotFoundError, match="モデルファイルが見つかりません"):
            PochiPredictor(
                model_name="resnet18",
                num_classes=3,
                device="cpu",
                model_path=str(tmp_path / "nonexistent.pth"),
            )

    def test_not_instance_of_trainer(self, tmp_path):
        """PochiPredictorはPochiTrainerのインスタンスではない."""
        checkpoint_path = _create_test_checkpoint(tmp_path)
        predictor = PochiPredictor(
            model_name="resnet18",
            num_classes=3,
            device="cpu",
            model_path=str(checkpoint_path),
        )
        assert not isinstance(predictor, PochiTrainer)


class TestPochiPredictorGetModelInfo:
    """get_model_infoメソッドのテスト."""

    def test_model_info_contents(self, tmp_path):
        """モデル情報に必要なキーが含まれる."""
        checkpoint_path = _create_test_checkpoint(tmp_path)
        predictor = PochiPredictor(
            model_name="resnet18",
            num_classes=3,
            device="cpu",
            model_path=str(checkpoint_path),
        )
        info = predictor.get_model_info()

        assert info["model_name"] == "resnet18"
        assert info["num_classes"] == 3
        assert "cpu" in info["device"]
        assert info["model_path"] == str(checkpoint_path)
        assert info["best_accuracy"] == 85.0
        assert info["epoch"] == 5
