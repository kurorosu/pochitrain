"""infer_onnx CLIのテスト.

ONNX推論CLIの引数パースと入口導線をテスト.
実際のONNX推論はtest_onnx/test_inference.pyでテスト済み.
"""

import argparse
from unittest.mock import patch

import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")


class TestInferOnnxArgumentParsing:
    """infer_onnx CLIの引数パースのテスト."""

    def _build_parser(self):
        """テスト用にinfer_onnxと同等のパーサーを構築."""
        parser = argparse.ArgumentParser(description="ONNXモデルを使用した推論")
        parser.add_argument("model_path", help="ONNXモデルファイルパス")
        parser.add_argument("--data", help="推論データディレクトリ")
        parser.add_argument("--output", "-o", help="結果出力ディレクトリ")
        return parser

    def test_model_path_required(self):
        """model_pathが必須引数であることを確認."""
        parser = self._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_model_path_parsed(self):
        """model_pathが正しくパースされる."""
        parser = self._build_parser()
        args = parser.parse_args(["model.onnx"])
        assert args.model_path == "model.onnx"

    def test_data_option(self):
        """--dataオプションが正しくパースされる."""
        parser = self._build_parser()
        args = parser.parse_args(["model.onnx", "--data", "data/val"])
        assert args.data == "data/val"

    def test_output_option(self):
        """--outputオプションが正しくパースされる."""
        parser = self._build_parser()
        args = parser.parse_args(["model.onnx", "--output", "results/"])
        assert args.output == "results/"

    def test_output_short_option(self):
        """-oオプションが正しくパースされる."""
        parser = self._build_parser()
        args = parser.parse_args(["model.onnx", "-o", "results/"])
        assert args.output == "results/"

    def test_data_default_none(self):
        """--data未指定時はNone."""
        parser = self._build_parser()
        args = parser.parse_args(["model.onnx"])
        assert args.data is None

    def test_output_default_none(self):
        """--output未指定時はNone."""
        parser = self._build_parser()
        args = parser.parse_args(["model.onnx"])
        assert args.output is None


class TestInferOnnxMainExit:
    """main関数のSystemExitテスト."""

    def test_main_no_args_exits(self):
        """引数なしでSystemExitが発生する."""
        from pochitrain.cli.infer_onnx import main

        with patch("sys.argv", ["infer-onnx"]):
            with pytest.raises(SystemExit):
                main()

    def test_main_nonexistent_model_exits(self, tmp_path):
        """存在しないモデルでSystemExitが発生する."""
        from pochitrain.cli.infer_onnx import main

        fake_model = str(tmp_path / "nonexistent.onnx")
        with patch("sys.argv", ["infer-onnx", fake_model]):
            with pytest.raises(SystemExit):
                main()


class TestGpuFallbackReresolution:
    """OnnxInference の GPU フォールバック後にパイプラインが再解決されるテスト.

    OnnxInference が内部で use_gpu=False に切り替えた場合,
    CLI側で pipeline と dataset を再解決するロジックを検証する.
    main() を通して実コード経路のフォールバックを検証する.
    """

    @staticmethod
    def _create_test_env(tmp_path):
        """テスト用のONNXモデル・config・データセットを作成する.

        Returns:
            (model_path, data_path, output_dir) のタプル
        """
        import torch
        import torch.nn as nn

        class _SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(16, 2)

            def forward(self, x):
                x = torch.relu(self.conv(x))
                x = self.pool(x)
                return self.fc(x.view(x.size(0), -1))

        # work_dir/models/model.onnx
        work_dir = tmp_path / "work_dir"
        models_dir = work_dir / "models"
        models_dir.mkdir(parents=True)

        model = _SimpleModel()
        model.eval()
        model_path = models_dir / "model.onnx"
        torch.onnx.export(
            model,
            torch.randn(1, 3, 32, 32),
            str(model_path),
            export_params=True,
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )

        # データセット: data/{class_a, class_b}/ にダミー画像 (モデルの2クラスに対応)
        from PIL import Image

        for cls_name, color in [("class_a", "red"), ("class_b", "blue")]:
            cls_dir = tmp_path / "data" / cls_name
            cls_dir.mkdir(parents=True)
            img = Image.new("RGB", (32, 32), color=color)
            img.save(str(cls_dir / "dummy.jpg"))

        # config.py: device="cuda" でGPU推論を要求
        config_path = work_dir / "config.py"
        config_path.write_text(
            "from torchvision import transforms\n"
            f'val_data_root = r"{tmp_path / "data"}"\n'
            "val_transform = transforms.Compose([\n"
            "    transforms.Resize((32, 32)),\n"
            "    transforms.ToTensor(),\n"
            "    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),\n"
            "])\n"
            'device = "cuda"\n'
            "batch_size = 1\n"
            "num_workers = 0\n",
            encoding="utf-8",
        )

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        data_path = tmp_path / "data"
        return model_path, data_path, output_dir

    def test_main_does_not_crash_when_cuda_ep_unavailable(self, tmp_path):
        """CUDA EP不可時にmain()がRuntimeErrorなく完走する."""
        from pochitrain.cli.infer_onnx import main

        model_path, data_path, output_dir = self._create_test_env(tmp_path)

        with (
            patch(
                "sys.argv",
                [
                    "infer-onnx",
                    str(model_path),
                    "--data",
                    str(data_path),
                    "-o",
                    str(output_dir),
                    "--pipeline",
                    "gpu",
                ],
            ),
            patch(
                "pochitrain.onnx.inference.check_gpu_availability",
                return_value=False,
            ),
        ):
            # RuntimeError が発生しないことを確認
            main()

        # 結果ファイルが生成されていることを確認
        csv_files = list(output_dir.glob("*.csv"))
        assert len(csv_files) > 0

    def test_main_fallback_uses_set_input_not_set_input_gpu(self, tmp_path):
        """フォールバック後にset_input_gpu()が呼ばれないことを確認."""
        from unittest.mock import MagicMock

        from pochitrain.cli.infer_onnx import main

        model_path, data_path, output_dir = self._create_test_env(tmp_path)

        with (
            patch(
                "sys.argv",
                [
                    "infer-onnx",
                    str(model_path),
                    "--data",
                    str(data_path),
                    "-o",
                    str(output_dir),
                    "--pipeline",
                    "gpu",
                ],
            ),
            patch(
                "pochitrain.onnx.inference.check_gpu_availability",
                return_value=False,
            ),
            patch(
                "pochitrain.onnx.inference.OnnxInference.set_input_gpu",
                side_effect=RuntimeError("set_input_gpuが呼ばれた"),
            ) as mock_set_input_gpu,
        ):
            main()

        mock_set_input_gpu.assert_not_called()
