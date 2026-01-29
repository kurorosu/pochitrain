"""PochiPredictorクラスのテスト.

実際のモデルとチェックポイントを使用した古典的テスト.
"""

import tempfile
from pathlib import Path

import pytest
import torch
import torchvision.transforms as transforms

from pochitrain.models.pochi_models import PochiModel
from pochitrain.pochi_predictor import PochiPredictor


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


def _create_test_dataset(
    tmp_path: Path, num_classes: int = 3, images_per_class: int = 2
) -> Path:
    """テスト用データセットを作成.

    Args:
        tmp_path: 一時ディレクトリ
        num_classes: クラス数
        images_per_class: クラスあたりの画像数

    Returns:
        データセットのルートパス
    """
    from PIL import Image

    data_root = tmp_path / "val_data"
    class_names = [f"class_{i}" for i in range(num_classes)]

    for class_name in class_names:
        class_dir = data_root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for j in range(images_per_class):
            img = Image.new("RGB", (32, 32), color=(j * 50, j * 30, j * 20))
            img.save(class_dir / f"img_{j}.jpg")

    return data_root


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
            work_dir=str(tmp_path / "inference_results"),
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
            work_dir=str(tmp_path / "inference_results"),
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
            work_dir=str(tmp_path / "inference_results"),
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
            work_dir=str(tmp_path / "inference_results"),
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
                work_dir=str(tmp_path / "inference_results"),
            )

    def test_inference_workspace_not_created_on_init(self, tmp_path):
        """初期化時に推論ワークスペースが作成されない（遅延作成）."""
        checkpoint_path = _create_test_checkpoint(tmp_path)
        predictor = PochiPredictor(
            model_name="resnet18",
            num_classes=3,
            device="cpu",
            model_path=str(checkpoint_path),
            work_dir=str(tmp_path / "inference_results"),
        )
        assert predictor.inference_workspace is None


class TestPochiPredictorCalculateAccuracy:
    """calculate_accuracyメソッドのテスト."""

    def _create_predictor(self, tmp_path):
        checkpoint_path = _create_test_checkpoint(tmp_path)
        return PochiPredictor(
            model_name="resnet18",
            num_classes=3,
            device="cpu",
            model_path=str(checkpoint_path),
            work_dir=str(tmp_path / "inference_results"),
        )

    def test_perfect_accuracy(self, tmp_path):
        """全問正解で100%."""
        predictor = self._create_predictor(tmp_path)
        result = predictor.calculate_accuracy([0, 1, 2], [0, 1, 2])
        assert result["accuracy_percentage"] == 100.0
        assert result["correct_predictions"] == 3
        assert result["total_samples"] == 3

    def test_zero_accuracy(self, tmp_path):
        """全問不正解で0%."""
        predictor = self._create_predictor(tmp_path)
        result = predictor.calculate_accuracy([1, 2, 0], [0, 1, 2])
        assert result["accuracy_percentage"] == 0.0
        assert result["correct_predictions"] == 0

    def test_partial_accuracy(self, tmp_path):
        """部分正解."""
        predictor = self._create_predictor(tmp_path)
        result = predictor.calculate_accuracy([0, 1, 0, 1], [0, 0, 0, 1])
        assert result["accuracy_percentage"] == 75.0
        assert result["correct_predictions"] == 3

    def test_empty_lists(self, tmp_path):
        """空のリストで0%."""
        predictor = self._create_predictor(tmp_path)
        result = predictor.calculate_accuracy([], [])
        assert result["accuracy_percentage"] == 0.0
        assert result["total_samples"] == 0


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
            work_dir=str(tmp_path / "inference_results"),
        )
        info = predictor.get_model_info()

        assert info["model_name"] == "resnet18"
        assert info["num_classes"] == 3
        assert "cpu" in info["device"]
        assert info["model_path"] == str(checkpoint_path)
        assert info["best_accuracy"] == 85.0
        assert info["epoch"] == 5


class TestPochiPredictorGetInferenceWorkspaceInfo:
    """get_inference_workspace_infoメソッドのテスト."""

    def test_no_workspace_created(self, tmp_path):
        """ワークスペース未作成時の情報."""
        checkpoint_path = _create_test_checkpoint(tmp_path)
        predictor = PochiPredictor(
            model_name="resnet18",
            num_classes=3,
            device="cpu",
            model_path=str(checkpoint_path),
            work_dir=str(tmp_path / "inference_results"),
        )
        info = predictor.get_inference_workspace_info()
        assert info["workspace_path"] is None
        assert info["exists"] is False


class TestPochiPredictorEnsureWorkspace:
    """_ensure_inference_workspaceメソッドのテスト."""

    def test_creates_workspace_on_first_call(self, tmp_path):
        """初回呼び出しでワークスペースが作成される."""
        checkpoint_path = _create_test_checkpoint(tmp_path)
        predictor = PochiPredictor(
            model_name="resnet18",
            num_classes=3,
            device="cpu",
            model_path=str(checkpoint_path),
            work_dir=str(tmp_path / "inference_results"),
        )
        assert predictor.inference_workspace is None

        workspace = predictor._ensure_inference_workspace()
        assert workspace is not None
        assert workspace.exists()
        assert predictor.inference_workspace == workspace

    def test_returns_same_workspace_on_subsequent_calls(self, tmp_path):
        """2回目以降は同じワークスペースを返す."""
        checkpoint_path = _create_test_checkpoint(tmp_path)
        predictor = PochiPredictor(
            model_name="resnet18",
            num_classes=3,
            device="cpu",
            model_path=str(checkpoint_path),
            work_dir=str(tmp_path / "inference_results"),
        )
        workspace1 = predictor._ensure_inference_workspace()
        workspace2 = predictor._ensure_inference_workspace()
        assert workspace1 == workspace2


class TestPochiPredictorPredictWithPaths:
    """predict_with_pathsメソッドのテスト."""

    def test_predict_returns_correct_structure(self, tmp_path):
        """predict_with_pathsが正しい構造のタプルを返す."""
        checkpoint_path = _create_test_checkpoint(tmp_path, num_classes=3)
        data_root = _create_test_dataset(tmp_path, num_classes=3, images_per_class=2)

        predictor = PochiPredictor(
            model_name="resnet18",
            num_classes=3,
            device="cpu",
            model_path=str(checkpoint_path),
            work_dir=str(tmp_path / "inference_results"),
        )

        image_paths, predicted_labels, true_labels, confidence_scores, class_names = (
            predictor.predict_with_paths(
                val_data_root=str(data_root),
                batch_size=2,
                num_workers=0,
                image_size=32,
            )
        )

        total_images = 3 * 2  # 3 classes * 2 images
        assert len(image_paths) == total_images
        assert len(predicted_labels) == total_images
        assert len(true_labels) == total_images
        assert len(confidence_scores) == total_images
        assert len(class_names) == 3

    def test_predicted_labels_in_range(self, tmp_path):
        """予測ラベルが有効な範囲内."""
        checkpoint_path = _create_test_checkpoint(tmp_path, num_classes=3)
        data_root = _create_test_dataset(tmp_path, num_classes=3, images_per_class=1)

        predictor = PochiPredictor(
            model_name="resnet18",
            num_classes=3,
            device="cpu",
            model_path=str(checkpoint_path),
            work_dir=str(tmp_path / "inference_results"),
        )

        _, predicted_labels, _, confidence_scores, _ = predictor.predict_with_paths(
            val_data_root=str(data_root),
            batch_size=1,
            num_workers=0,
            image_size=32,
        )

        for label in predicted_labels:
            assert 0 <= label < 3

        for conf in confidence_scores:
            assert 0.0 <= conf <= 1.0
