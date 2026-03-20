"""エッジケーステスト.

既存テストで不足している境界値・異常系テストを追加する.
対象: early_stopping, pochi_dataset, training_configurator, checkpoint_store, evaluator.
"""

import logging
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from pochitrain.models.pochi_models import create_model
from pochitrain.pochi_dataset import PochiImageDataset
from pochitrain.training.checkpoint_store import CheckpointStore
from pochitrain.training.early_stopping import EarlyStopping
from pochitrain.training.evaluator import Evaluator
from pochitrain.training.layer_wise_lr import ParamGroupBuilder, ResNetLayerGrouper
from pochitrain.training.training_configurator import TrainingConfigurator

# --- EarlyStopping エッジケース ---


class TestEarlyStoppingMinDeltaEdgeCases:
    """EarlyStopping の min_delta 境界値テスト."""

    def test_very_small_min_delta_detects_tiny_improvement(self):
        """極小 min_delta (0.0001) で微小な改善を検出する."""
        es = EarlyStopping(patience=3, min_delta=0.0001, monitor="val_accuracy")
        es.step(90.0, 1)
        # 0.0002 > min_delta(0.0001) なので改善と判定
        es.step(90.0002, 2)

        assert es.counter == 0
        assert es.best_value == pytest.approx(90.0002)

    def test_very_small_min_delta_rejects_insufficient_improvement(self):
        """極小 min_delta (0.0001) で不十分な改善を拒否する."""
        es = EarlyStopping(patience=2, min_delta=0.0001, monitor="val_accuracy")
        es.step(90.0, 1)
        # 0.00005 < min_delta(0.0001) なので改善なし
        es.step(90.00005, 2)

        assert es.counter == 1

    def test_large_min_delta_requires_significant_improvement(self):
        """極大 min_delta (10.0) で大幅な改善のみ検出する."""
        es = EarlyStopping(patience=2, min_delta=10.0, monitor="val_accuracy")
        es.step(50.0, 1)
        # +9.0 < min_delta(10.0) なので改善なし
        es.step(59.0, 2)

        assert es.counter == 1

    def test_large_min_delta_accepts_sufficient_improvement(self):
        """極大 min_delta (10.0) で十分な改善を検出する."""
        es = EarlyStopping(patience=2, min_delta=10.0, monitor="val_accuracy")
        es.step(50.0, 1)
        # +11.0 > min_delta(10.0) なので改善
        es.step(61.0, 2)

        assert es.counter == 0
        assert es.best_value == pytest.approx(61.0)

    def test_large_min_delta_with_val_loss(self):
        """極大 min_delta (10.0) で val_loss 監視時の境界値テスト."""
        es = EarlyStopping(patience=2, min_delta=10.0, monitor="val_loss")
        es.step(50.0, 1)
        # -9.0 < min_delta(10.0) なので改善なし
        es.step(41.0, 2)

        assert es.counter == 1

    def test_large_min_delta_val_loss_accepts_sufficient_decrease(self):
        """極大 min_delta (10.0) で val_loss の十分な減少を検出する."""
        es = EarlyStopping(patience=2, min_delta=10.0, monitor="val_loss")
        es.step(50.0, 1)
        # -11.0 > min_delta(10.0) なので改善
        es.step(39.0, 2)

        assert es.counter == 0
        assert es.best_value == pytest.approx(39.0)


# --- PochiImageDataset エッジケース ---


class TestPochiDatasetEdgeCases:
    """PochiImageDataset の境界値テスト."""

    def test_single_image_dataset(self, create_dummy_dataset):
        """画像1枚のみのデータセットが正常に動作する."""
        dataset_path = create_dummy_dataset({"single_class": 1})

        dataset = PochiImageDataset(str(dataset_path))

        assert len(dataset) == 1
        image, label = dataset[0]
        assert image is not None
        assert label == 0

    def test_single_image_per_class_multiple_classes(self, create_dummy_dataset):
        """各クラス1枚ずつの複数クラスデータセット."""
        dataset_path = create_dummy_dataset({"a": 1, "b": 1, "c": 1})

        dataset = PochiImageDataset(str(dataset_path))

        assert len(dataset) == 3
        assert len(dataset.classes) == 3

    def test_corrupted_image_file(self, tmp_path: Path):
        """破損した画像ファイルがデータセット読み込み時にエラーを起こす."""
        class_dir = tmp_path / "broken_class"
        class_dir.mkdir()

        # 正常な画像を1枚作成 (データセット作成用)
        img = Image.new("RGB", (32, 32), color=(100, 100, 100))
        img.save(class_dir / "good.jpg")

        # 破損ファイルを作成
        (class_dir / "corrupted.jpg").write_bytes(b"this is not a valid image")

        dataset = PochiImageDataset(str(tmp_path))
        # データセット作成は成功する (ファイル一覧取得のみ)
        assert len(dataset) == 2

        # 破損画像へのアクセスで例外が発生する
        errors = 0
        for i in range(len(dataset)):
            try:
                dataset[i]
            except Exception:
                errors += 1

        assert errors >= 1


# --- TrainingConfigurator エッジケース ---


class TestTrainingConfiguratorEdgeCases:
    """TrainingConfigurator の境界値テスト."""

    @pytest.fixture
    def configurator(self):
        """テスト用コンフィギュレータ."""
        logger = logging.getLogger("test_edge")
        return TrainingConfigurator(
            device=torch.device("cpu"),
            logger=logger,
            param_group_builder=ParamGroupBuilder(ResNetLayerGrouper(), logger),
        )

    @pytest.fixture
    def model(self):
        """テスト用モデル."""
        return create_model("resnet18", num_classes=4, pretrained=False)

    def test_zero_learning_rate(self, configurator, model):
        """学習率0.0でもオプティマイザが構築できる."""
        components = configurator.configure(
            model=model,
            learning_rate=0.0,
            optimizer_name="Adam",
        )

        assert components.base_learning_rate == 0.0
        # lr=0 ではパラメータが更新されないことを確認
        params_before = [p.clone() for p in model.parameters()]
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)
        loss = output.sum()
        loss.backward()
        components.optimizer.step()

        for before, after in zip(params_before, model.parameters()):
            assert torch.equal(before, after.data)

    def test_negative_learning_rate_accepted_by_pytorch(self, configurator, model):
        """負の学習率でもオプティマイザ構築は成功する (PyTorch の仕様)."""
        components = configurator.configure(
            model=model,
            learning_rate=-0.001,
            optimizer_name="SGD",
        )

        assert components.base_learning_rate == -0.001

    def test_extreme_layer_wise_lr_high(self, configurator, model):
        """極端に高い層別学習率でも構築可能."""
        layer_wise_lr_config = {
            "layer_rates": {
                "fc": 100.0,
            },
        }

        components = configurator.configure(
            model=model,
            learning_rate=0.001,
            optimizer_name="SGD",
            enable_layer_wise_lr=True,
            layer_wise_lr_config=layer_wise_lr_config,
        )

        fc_groups = [
            g for g in components.optimizer.param_groups if g["layer_name"] == "fc"
        ]
        assert len(fc_groups) == 1
        assert fc_groups[0]["lr"] == pytest.approx(100.0)

    def test_extreme_layer_wise_lr_zero(self, configurator, model):
        """層別学習率0.0 (凍結相当) でも構築可能."""
        layer_wise_lr_config = {
            "layer_rates": {
                "conv1": 0.0,
                "layer1": 0.0,
            },
        }

        components = configurator.configure(
            model=model,
            learning_rate=0.001,
            optimizer_name="SGD",
            enable_layer_wise_lr=True,
            layer_wise_lr_config=layer_wise_lr_config,
        )

        frozen_groups = [
            g
            for g in components.optimizer.param_groups
            if g["layer_name"] in ("conv1", "layer1")
        ]
        for g in frozen_groups:
            assert g["lr"] == 0.0


# --- CheckpointStore エッジケース ---


class TestCheckpointStoreEdgeCases:
    """CheckpointStore の境界値テスト."""

    def test_save_to_nonexistent_directory(self, tmp_path: Path):
        """存在しないディレクトリへの保存でエラーが発生する."""
        nonexistent_dir = tmp_path / "does" / "not" / "exist"

        logger = logging.getLogger("test_cp")
        store = CheckpointStore(nonexistent_dir, logger)
        model = nn.Linear(10, 2)
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        with pytest.raises(Exception):
            store.save_best_model(
                epoch=1,
                model=model,
                optimizer=optimizer,
                scheduler=None,
                best_accuracy=50.0,
            )

    def test_save_last_model_without_optimizer(self, tmp_path: Path):
        """optimizer=None でもラストモデル保存可能."""
        logger = logging.getLogger("test_cp")
        store = CheckpointStore(tmp_path, logger)
        model = nn.Linear(10, 2)

        store.save_last_model(
            epoch=1,
            model=model,
            optimizer=None,
            scheduler=None,
            best_accuracy=0.0,
        )

        path = tmp_path / "last_model.pth"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        assert checkpoint["optimizer_state_dict"] is None


# --- Evaluator エッジケース ---


class TestEvaluatorNanInfHandling:
    """Evaluator の NaN/Inf 値処理テスト."""

    @pytest.fixture
    def evaluator(self):
        """CPU 上の Evaluator インスタンス."""
        logger = logging.getLogger("test_eval_edge")
        return Evaluator(device=torch.device("cpu"), logger=logger)

    def test_validate_empty_loader(self, evaluator):
        """空の DataLoader で防御的ガードが機能する."""
        model = nn.Linear(4, 3)
        criterion = nn.CrossEntropyLoss()
        empty_dataset = TensorDataset(
            torch.empty(0, 4), torch.empty(0, dtype=torch.long)
        )
        loader = DataLoader(empty_dataset, batch_size=1)

        result = evaluator.validate(model, loader, criterion)

        assert result["val_loss"] == 0.0
        assert result["val_accuracy"] == 0.0

    def test_calculate_accuracy_empty_lists(self, evaluator):
        """空リストで精度0%を返す."""
        result = evaluator.calculate_accuracy([], [])

        assert result["accuracy_percentage"] == 0.0
        assert result["total_samples"] == 0
        assert result["correct_predictions"] == 0

    def test_calculate_accuracy_single_sample(self, evaluator):
        """サンプル1つの場合の精度計算."""
        result = evaluator.calculate_accuracy([0], [0])

        assert result["accuracy_percentage"] == 100.0
        assert result["total_samples"] == 1

    def test_confusion_matrix_single_sample(self, evaluator):
        """サンプル1つの混同行列."""
        predicted = torch.tensor([1])
        targets = torch.tensor([1])

        cm = evaluator._compute_confusion_matrix(predicted, targets, 3)

        assert cm[1, 1].item() == 1
        assert cm.sum().item() == 1
