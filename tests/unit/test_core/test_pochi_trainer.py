"""
PochiTrainerの基本テスト
"""

import tempfile

import pytest
import torch

from pochitrain.pochi_trainer import PochiTrainer


def test_pochi_trainer_init():
    """PochiTrainerの初期化テスト"""
    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = PochiTrainer(
            model_name="resnet18",
            num_classes=10,
            pretrained=False,  # テスト用に高速化
            device="cpu",  # テスト用にCPUを指定
            work_dir=temp_dir,  # テスト用に一時ディレクトリを使用
        )

        assert trainer.model is not None
        assert trainer.device == torch.device("cpu")
        assert trainer.epoch == 0
        assert trainer.best_accuracy == 0.0


def test_pochi_trainer_cudnn_benchmark_disabled_on_cpu():
    """CPU環境ではcudnn_benchmarkが無効になることをテスト"""
    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = PochiTrainer(
            model_name="resnet18",
            num_classes=10,
            pretrained=False,
            device="cpu",
            work_dir=temp_dir,
            cudnn_benchmark=True,  # TrueでもCPUでは無効
        )

        assert trainer.cudnn_benchmark is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_pochi_trainer_cudnn_benchmark_enabled_on_cuda():
    """CUDA環境でcudnn_benchmark=Trueが有効になることをテスト"""
    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = PochiTrainer(
            model_name="resnet18",
            num_classes=10,
            pretrained=False,
            device="cuda",
            work_dir=temp_dir,
            cudnn_benchmark=True,
        )

        assert trainer.cudnn_benchmark is True
        assert torch.backends.cudnn.benchmark is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_pochi_trainer_cudnn_benchmark_disabled_on_cuda():
    """CUDA環境でcudnn_benchmark=Falseが無効のままであることをテスト"""
    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = PochiTrainer(
            model_name="resnet18",
            num_classes=10,
            pretrained=False,
            device="cuda",
            work_dir=temp_dir,
            cudnn_benchmark=False,
        )

        assert trainer.cudnn_benchmark is False


def test_pochi_trainer_setup_training():
    """PochiTrainerの訓練設定テスト"""
    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = PochiTrainer(
            model_name="resnet18",
            num_classes=10,
            pretrained=False,
            device="cpu",
            work_dir=temp_dir,
        )

        trainer.setup_training(learning_rate=0.001, optimizer_name="Adam")

        assert trainer.optimizer is not None
        assert trainer.criterion is not None


def test_pochi_trainer_invalid_model():
    """無効なモデル名のテスト"""
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(ValueError):
            PochiTrainer(
                model_name="invalid_model",
                num_classes=10,
                pretrained=False,
                device="cpu",
                work_dir=temp_dir,
            )


def test_pochi_trainer_invalid_optimizer():
    """無効な最適化器のテスト"""
    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = PochiTrainer(
            model_name="resnet18",
            num_classes=10,
            pretrained=False,
            device="cpu",
            work_dir=temp_dir,
        )

        with pytest.raises(ValueError):
            trainer.setup_training(
                learning_rate=0.001, optimizer_name="InvalidOptimizer"
            )


def test_pochi_trainer_class_weights_cpu():
    """CPU環境でのクラス重み設定テスト"""
    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = PochiTrainer(
            model_name="resnet18",
            num_classes=3,
            pretrained=False,
            device="cpu",
            work_dir=temp_dir,
        )

        # クラス重みを設定
        class_weights = [1.0, 2.0, 0.5]
        trainer.setup_training(
            learning_rate=0.001,
            optimizer_name="Adam",
            class_weights=class_weights,
            num_classes=3,
        )

        assert trainer.criterion is not None
        # 損失関数の重みがCPUデバイスに配置されていることを確認
        assert trainer.criterion.weight.device == torch.device("cpu")
        # 重みの値が正しく設定されていることを確認
        expected_weights = torch.tensor(class_weights, dtype=torch.float32)
        assert torch.allclose(trainer.criterion.weight, expected_weights)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_pochi_trainer_class_weights_cuda():
    """CUDA環境でのクラス重み設定テスト（CUDA利用可能時のみ）"""
    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = PochiTrainer(
            model_name="resnet18",
            num_classes=4,
            pretrained=False,
            device="cuda",
            work_dir=temp_dir,
        )

        # クラス重みを設定
        class_weights = [1.0, 3.0, 1.5, 0.8]
        trainer.setup_training(
            learning_rate=0.001,
            optimizer_name="Adam",
            class_weights=class_weights,
            num_classes=4,
        )

        assert trainer.criterion is not None
        # 損失関数の重みがCUDAデバイスに配置されていることを確認
        assert trainer.criterion.weight.device.type == "cuda"
        # 重みの値が正しく設定されていることを確認
        expected_weights = torch.tensor(class_weights, dtype=torch.float32)
        assert torch.allclose(trainer.criterion.weight.cpu(), expected_weights)


def test_pochi_trainer_class_weights_mismatch():
    """クラス重みとクラス数の不整合テスト"""
    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = PochiTrainer(
            model_name="resnet18",
            num_classes=3,
            pretrained=False,
            device="cpu",
            work_dir=temp_dir,
        )

        # クラス重みの長さがクラス数と一致しない場合
        class_weights = [1.0, 2.0]  # 2つの重みだが、クラス数は3
        with pytest.raises(
            ValueError, match="クラス重みの長さ.*がクラス数.*と一致しません"
        ):
            trainer.setup_training(
                learning_rate=0.001,
                optimizer_name="Adam",
                class_weights=class_weights,
                num_classes=3,
            )


def test_pochi_trainer_class_weights_none():
    """クラス重みなしの場合のテスト"""
    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = PochiTrainer(
            model_name="resnet18",
            num_classes=5,
            pretrained=False,
            device="cpu",
            work_dir=temp_dir,
        )

        trainer.setup_training(
            learning_rate=0.001,
            optimizer_name="Adam",
            class_weights=None,
        )

        assert trainer.criterion is not None
        # 重みが設定されていないことを確認
        assert trainer.criterion.weight is None


class TestConfusionMatrixCalculation:
    """混同行列計算メソッドのテストクラス"""

    def setup_method(self):
        """各テストメソッドの前に実行される setup"""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = PochiTrainer(
                model_name="resnet18",
                num_classes=4,
                pretrained=False,
                device="cpu",
                work_dir=temp_dir,
            )
            # 混同行列計算用のクラス数を設定
            self.trainer.num_classes_for_cm = 4

    def test_confusion_matrix_basic_case(self):
        """基本的な混同行列計算のテスト"""
        # 予測値と正解値を作成（4クラス分類）
        predicted = torch.tensor([0, 1, 2, 3, 0, 1])
        targets = torch.tensor([0, 1, 2, 3, 0, 2])

        # 混同行列を計算
        cm = self.trainer._compute_confusion_matrix_pytorch(predicted, targets, 4)

        # 期待される混同行列
        # 正解0→予測0: 2回, 正解2→予測1: 1回, 正解2→予測2: 1回, その他0回
        expected = torch.tensor(
            [
                [2, 0, 0, 0],  # 正解0: 予測0が2回
                [0, 1, 0, 0],  # 正解1: 予測1が1回
                [0, 1, 1, 0],  # 正解2: 予測1が1回、予測2が1回
                [0, 0, 0, 1],  # 正解3: 予測3が1回
            ]
        )

        assert torch.equal(cm, expected), f"Expected {expected}, but got {cm}"

    def test_confusion_matrix_perfect_prediction(self):
        """完全に正解した場合の混同行列テスト"""
        # 全て正解のケース
        predicted = torch.tensor([0, 1, 2, 3])
        targets = torch.tensor([0, 1, 2, 3])

        cm = self.trainer._compute_confusion_matrix_pytorch(predicted, targets, 4)

        # 対角成分のみ1で、他は0の単位行列になるはず
        expected = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        assert torch.equal(cm, expected)

    def test_confusion_matrix_all_wrong(self):
        """全て間違った場合の混同行列テスト"""
        # 全て1つずれて間違っているケース
        predicted = torch.tensor([1, 2, 3, 0])
        targets = torch.tensor([0, 1, 2, 3])

        cm = self.trainer._compute_confusion_matrix_pytorch(predicted, targets, 4)

        # 対角成分が0で、1つずれた位置に1が入るはず
        expected = torch.tensor(
            [
                [0, 1, 0, 0],  # 正解0→予測1
                [0, 0, 1, 0],  # 正解1→予測2
                [0, 0, 0, 1],  # 正解2→予測3
                [1, 0, 0, 0],  # 正解3→予測0
            ]
        )

        assert torch.equal(cm, expected)

    def test_confusion_matrix_empty_input(self):
        """空の入力に対するテスト"""
        predicted = torch.tensor([])
        targets = torch.tensor([])

        cm = self.trainer._compute_confusion_matrix_pytorch(predicted, targets, 4)

        # 全て0の行列になるはず
        expected = torch.zeros(4, 4, dtype=torch.int64)
        assert torch.equal(cm, expected)

    def test_confusion_matrix_single_class(self):
        """単一クラスのみの場合のテスト"""
        # 全てクラス2のケース
        predicted = torch.tensor([2, 2, 2])
        targets = torch.tensor([2, 2, 2])

        cm = self.trainer._compute_confusion_matrix_pytorch(predicted, targets, 4)

        expected = torch.tensor(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 3, 0],  # クラス2→クラス2が3回
                [0, 0, 0, 0],
            ]
        )

        assert torch.equal(cm, expected)

    def test_confusion_matrix_device_consistency(self):
        """デバイス間での一貫性テスト"""
        # CPU上でテンソルを作成
        predicted = torch.tensor([0, 1, 2, 1])
        targets = torch.tensor([0, 1, 1, 2])

        cm = self.trainer._compute_confusion_matrix_pytorch(predicted, targets, 3)

        # 結果がCPU上にあることを確認
        assert cm.device.type == "cpu"

        # 期待される結果
        expected = torch.tensor(
            [
                [1, 0, 0],  # 正解0→予測0: 1回
                [0, 1, 1],  # 正解1→予測1: 1回, 正解1→予測2: 1回
                [0, 1, 0],  # 正解2→予測1: 1回
            ]
        )

        assert torch.equal(cm, expected)
