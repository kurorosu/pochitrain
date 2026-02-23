"""推論パイプライン整合性テスト."""

from pathlib import Path

import pytest
import torchvision.transforms as transforms
from PIL import Image

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from pochitrain.inference.pipeline_strategy import create_dataset_and_params
from pochitrain.inference.services.onnx_inference_service import OnnxInferenceService
from pochitrain.inference.services.trt_inference_service import TensorRTInferenceService
from pochitrain.pochi_dataset import (
    FastInferenceDataset,
    GpuInferenceDataset,
    PochiImageDataset,
)


def _make_data_root(tmp_path: Path) -> Path:
    """最小構成のクラスフォルダデータセットを作成する."""
    root = tmp_path / "data"
    for class_name, color in [("class_a", "red"), ("class_b", "blue")]:
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (32, 32), color=color)
        img.save(class_dir / "dummy.jpg")
    return root


def _valid_transform() -> transforms.Compose:
    """Normalize を含むテンソル互換 transform を返す."""
    return transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def _pil_only_transform() -> transforms.Compose:
    """PIL 専用処理を含む transform を返す."""
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )


def _no_normalize_transform() -> transforms.Compose:
    """Normalize を含まないテンソル互換 transform を返す."""
    return transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])


class TestPipelineConsistency:
    """データセット生成とパイプライン解決の整合性テスト."""

    def test_current_pipeline_same_result(self, tmp_path: Path) -> None:
        """current は ONNX/TRT で同一結果になる."""
        data_root = _make_data_root(tmp_path)
        transform = _valid_transform()

        onnx_ds, onnx_pipe, onnx_mean, onnx_std = create_dataset_and_params(
            "current", data_root, transform
        )
        trt_ds, trt_pipe, trt_mean, trt_std = create_dataset_and_params(
            "current", data_root, transform
        )

        assert isinstance(onnx_ds, PochiImageDataset)
        assert isinstance(trt_ds, PochiImageDataset)
        assert onnx_pipe == trt_pipe == "current"
        assert onnx_mean is trt_mean is None
        assert onnx_std is trt_std is None

    def test_fast_pipeline_same_result(self, tmp_path: Path) -> None:
        """fast は ONNX/TRT で同一結果になる."""
        data_root = _make_data_root(tmp_path)
        transform = _valid_transform()

        onnx_ds, onnx_pipe, _, _ = create_dataset_and_params(
            "fast", data_root, transform
        )
        trt_ds, trt_pipe, _, _ = create_dataset_and_params("fast", data_root, transform)

        assert isinstance(onnx_ds, FastInferenceDataset)
        assert isinstance(trt_ds, FastInferenceDataset)
        assert onnx_pipe == trt_pipe == "fast"

    def test_fast_with_pil_only_fallback_is_same(self, tmp_path: Path) -> None:
        """fast + PIL専用 transform は両CLIで current へフォールバックする."""
        data_root = _make_data_root(tmp_path)
        transform = _pil_only_transform()

        onnx_ds, onnx_pipe, _, _ = create_dataset_and_params(
            "fast", data_root, transform
        )
        trt_ds, trt_pipe, _, _ = create_dataset_and_params("fast", data_root, transform)

        assert isinstance(onnx_ds, PochiImageDataset)
        assert isinstance(trt_ds, PochiImageDataset)
        assert onnx_pipe == trt_pipe == "current"

    def test_gpu_without_normalize_fallback_is_same(self, tmp_path: Path) -> None:
        """gpu + Normalizeなし は両CLIで fast へフォールバックする."""
        data_root = _make_data_root(tmp_path)
        transform = _no_normalize_transform()

        onnx_ds, onnx_pipe, onnx_mean, onnx_std = create_dataset_and_params(
            "gpu", data_root, transform
        )
        trt_ds, trt_pipe, trt_mean, trt_std = create_dataset_and_params(
            "gpu", data_root, transform
        )

        assert isinstance(onnx_ds, FastInferenceDataset)
        assert isinstance(trt_ds, FastInferenceDataset)
        assert onnx_pipe == trt_pipe == "fast"
        assert onnx_mean is trt_mean is None
        assert onnx_std is trt_std is None

    def test_gpu_with_valid_transform_is_same(self, tmp_path: Path) -> None:
        """gpu + 正常 transform は両CLIで GpuInferenceDataset を使う."""
        data_root = _make_data_root(tmp_path)
        transform = _valid_transform()

        onnx_ds, onnx_pipe, onnx_mean, onnx_std = create_dataset_and_params(
            "gpu", data_root, transform
        )
        trt_ds, trt_pipe, trt_mean, trt_std = create_dataset_and_params(
            "gpu", data_root, transform
        )

        assert isinstance(onnx_ds, GpuInferenceDataset)
        assert isinstance(trt_ds, GpuInferenceDataset)
        assert onnx_pipe == trt_pipe == "gpu"
        assert onnx_mean == trt_mean == [0.485, 0.456, 0.406]
        assert onnx_std == trt_std == [0.229, 0.224, 0.225]


class TestAutoResolutionDifference:
    """ONNX/TRT 間の auto 解決仕様差を確認するテスト."""

    def test_auto_behavior_difference_is_intended(self) -> None:
        """ONNXのautoはuse_gpu依存, TRTのautoは常にgpu."""
        onnx_service = OnnxInferenceService()
        trt_service = TensorRTInferenceService()

        onnx_auto_cpu = onnx_service.resolve_pipeline("auto", use_gpu=False)
        onnx_auto_gpu = onnx_service.resolve_pipeline("auto", use_gpu=True)
        trt_auto = trt_service.resolve_pipeline("auto", use_gpu=True)

        assert onnx_auto_cpu == "fast"
        assert onnx_auto_gpu == "gpu"
        assert trt_auto == "gpu"
