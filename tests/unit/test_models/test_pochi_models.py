"""PochiModelクラスとcreate_model関数のテスト.

実際のモデルを作成してforward passまで実行する古典的テスト.
"""

import pytest
import torch
import torch.nn as nn

from pochitrain.models.pochi_models import PochiModel, create_model


class TestPochiModelInit:
    """PochiModel初期化のテスト."""

    @pytest.mark.parametrize("model_name", ["resnet18", "resnet34", "resnet50"])
    def test_supported_models(self, model_name):
        """サポートされているモデルが正常に作成される."""
        model = PochiModel(model_name, num_classes=10, pretrained=False)
        assert model.model_name == model_name
        assert model.num_classes == 10

    def test_unsupported_model_raises(self):
        """サポートされていないモデル名でValueErrorが発生する."""
        with pytest.raises(ValueError, match="サポートされていないモデル"):
            PochiModel("resnet101", num_classes=10, pretrained=False)

    def test_num_classes_applied(self):
        """num_classesが最終層に反映される."""
        model = PochiModel("resnet18", num_classes=5, pretrained=False)
        # ResNetのfc層の出力次元を確認
        assert model.model.fc.out_features == 5

    def test_pretrained_false(self):
        """pretrained=Falseでモデルが作成される."""
        model = PochiModel("resnet18", num_classes=3, pretrained=False)
        assert model is not None

    def test_is_nn_module(self):
        """nn.Moduleのサブクラスであることを確認."""
        model = PochiModel("resnet18", num_classes=5, pretrained=False)
        assert isinstance(model, nn.Module)


class TestPochiModelForward:
    """PochiModel.forward()のテスト."""

    def test_forward_output_shape(self):
        """forward passの出力shape確認."""
        model = PochiModel("resnet18", num_classes=5, pretrained=False)
        model.eval()

        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 5)

    def test_forward_different_num_classes(self):
        """異なるクラス数でのforward pass."""
        model = PochiModel("resnet18", num_classes=100, pretrained=False)
        model.eval()

        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 100)

    @pytest.mark.parametrize("model_name", ["resnet18", "resnet34", "resnet50"])
    def test_forward_all_models(self, model_name):
        """全モデルでforward passが成功する."""
        model = PochiModel(model_name, num_classes=10, pretrained=False)
        model.eval()

        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 10)

    def test_forward_small_image(self):
        """小さい画像でもforward passが成功する."""
        model = PochiModel("resnet18", num_classes=3, pretrained=False)
        model.eval()

        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 3)


class TestPochiModelGetModelInfo:
    """PochiModel.get_model_info()のテスト."""

    def test_info_keys(self):
        """必要なキーが含まれる."""
        model = PochiModel("resnet18", num_classes=5, pretrained=False)
        info = model.get_model_info()

        assert "model_name" in info
        assert "num_classes" in info
        assert "total_params" in info
        assert "trainable_params" in info

    def test_info_values(self):
        """情報の値が正しい."""
        model = PochiModel("resnet18", num_classes=5, pretrained=False)
        info = model.get_model_info()

        assert info["model_name"] == "resnet18"
        assert info["num_classes"] == 5
        assert info["total_params"] > 0
        assert info["trainable_params"] > 0

    def test_total_params_ge_trainable(self):
        """総パラメータ数 >= 訓練可能パラメータ数."""
        model = PochiModel("resnet18", num_classes=5, pretrained=False)
        info = model.get_model_info()
        assert info["total_params"] >= info["trainable_params"]

    def test_params_differ_by_num_classes(self):
        """クラス数が違えばパラメータ数が異なる."""
        model_5 = PochiModel("resnet18", num_classes=5, pretrained=False)
        model_100 = PochiModel("resnet18", num_classes=100, pretrained=False)

        info_5 = model_5.get_model_info()
        info_100 = model_100.get_model_info()

        assert info_5["total_params"] != info_100["total_params"]

    def test_params_differ_by_model(self):
        """モデルが違えばパラメータ数が異なる."""
        info_18 = PochiModel(
            "resnet18", num_classes=10, pretrained=False
        ).get_model_info()
        info_50 = PochiModel(
            "resnet50", num_classes=10, pretrained=False
        ).get_model_info()

        assert info_18["total_params"] < info_50["total_params"]


class TestCreateModel:
    """create_model便利関数のテスト."""

    def test_returns_pochi_model(self):
        """PochiModel型を返す."""
        model = create_model("resnet18", num_classes=5, pretrained=False)
        assert isinstance(model, PochiModel)

    def test_default_pretrained(self):
        """pretrained引数のデフォルトがTrueであることを確認."""
        # pretrained=Trueでもエラーなく作成できることだけ確認
        model = create_model("resnet18", num_classes=5)
        assert model is not None

    @pytest.mark.parametrize("model_name", ["resnet18", "resnet34", "resnet50"])
    def test_all_supported_models(self, model_name):
        """全サポートモデルが作成できる."""
        model = create_model(model_name, num_classes=3, pretrained=False)
        assert model.model_name == model_name
