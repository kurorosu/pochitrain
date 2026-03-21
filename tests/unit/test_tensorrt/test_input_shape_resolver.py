"""InputShapeResolver のテスト.

実際の ONNX モデルファイルを作成して入力形状解析を検証する古典派テスト.
"""

import logging
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from pochitrain.tensorrt.input_shape_resolver import InputShapeResolver

onnx = pytest.importorskip("onnx")


def _export_static_onnx(path: Path, input_h: int = 224, input_w: int = 224) -> Path:
    """静的シェイプの ONNX モデルを生成する."""

    class TinyModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.mean(dim=(2, 3))

    model = TinyModel()
    dummy = torch.randn(1, 3, input_h, input_w)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, (dummy,), str(path), input_names=["input"])
    return path


def _export_dynamic_onnx(path: Path) -> Path:
    """動的シェイプの ONNX モデルを生成する."""

    class TinyModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.mean(dim=(2, 3))

    model = TinyModel()
    dummy = torch.randn(1, 3, 224, 224)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy,),
        str(path),
        input_names=["input"],
        dynamic_axes={"input": {2: "height", 3: "width"}},
    )
    return path


class TestResolveWithCliInputSize:
    """CLI 引数で入力サイズが指定された場合のテスト."""

    def test_returns_chw_tuple(self):
        """CLI 入力 [H, W] が (3, H, W) に変換される."""
        resolver = InputShapeResolver()

        result = resolver.resolve([224, 224], Path("dummy.onnx"))

        assert result == (3, 224, 224)

    def test_non_square_input(self):
        """非正方形の入力サイズが正しく変換される."""
        resolver = InputShapeResolver()

        result = resolver.resolve([320, 480], Path("dummy.onnx"))

        assert result == (3, 320, 480)


class TestResolveWithStaticOnnx:
    """静的シェイプ ONNX モデルの場合のテスト."""

    def test_static_onnx_returns_none(self, tmp_path: Path):
        """静的シェイプの ONNX では None を返す (入力サイズ指定不要)."""
        onnx_path = _export_static_onnx(tmp_path / "static.onnx")
        resolver = InputShapeResolver()

        result = resolver.resolve(None, onnx_path)

        assert result is None


class TestResolveWithDynamicOnnx:
    """動的シェイプ ONNX モデルの場合のテスト."""

    def test_dynamic_onnx_raises_value_error(self, tmp_path: Path):
        """動的シェイプ検出時に ValueError が発生する."""
        onnx_path = _export_dynamic_onnx(tmp_path / "dynamic.onnx")
        resolver = InputShapeResolver()

        with pytest.raises(ValueError, match="動的シェイプ"):
            resolver.resolve(None, onnx_path)

    def test_dynamic_onnx_with_cli_input_succeeds(self, tmp_path: Path):
        """動的シェイプでも CLI 入力があれば解決できる."""
        onnx_path = _export_dynamic_onnx(tmp_path / "dynamic.onnx")
        resolver = InputShapeResolver()

        result = resolver.resolve([224, 224], onnx_path)

        assert result == (3, 224, 224)


class TestResolveWithoutOnnxPackage:
    """onnx パッケージが利用できない場合のテスト."""

    def test_returns_none_when_onnx_unavailable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """onnx が import できない場合は None を返す."""
        import builtins

        import pochitrain.tensorrt.input_shape_resolver as mod

        original_import = builtins.__import__

        from typing import Any, Mapping, Sequence

        def _fake_import(
            name: str,
            globals: Mapping[str, Any] | None = None,
            locals: Mapping[str, Any] | None = None,
            fromlist: Sequence[str] = (),
            level: int = 0,
        ) -> object:
            if name == "onnx":
                raise ImportError("no onnx")
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        resolver = mod.InputShapeResolver()
        result = resolver._detect_from_onnx(tmp_path / "any.onnx")

        assert result is None


class TestResolveWithCorruptedFile:
    """破損した ONNX ファイルの場合のテスト."""

    def test_corrupted_file_returns_none(self, tmp_path: Path):
        """破損ファイルでは None を返し, 例外を握りつぶさずログに出す."""
        bad_onnx = tmp_path / "corrupted.onnx"
        bad_onnx.write_bytes(b"not an onnx file")
        logger = logging.getLogger("test_corrupted")
        resolver = InputShapeResolver(logger)

        result = resolver.resolve(None, bad_onnx)

        assert result is None


class TestExtractStaticShape:
    """extract_static_shape のテスト."""

    def test_extracts_shape_from_static_onnx(self, tmp_path: Path):
        """静的 ONNX から (C, H, W) を取得できる."""
        onnx_path = _export_static_onnx(tmp_path / "static.onnx", 128, 256)
        resolver = InputShapeResolver()

        result = resolver.extract_static_shape(onnx_path)

        assert result == (3, 128, 256)

    def test_raises_runtime_error_for_invalid_file(self, tmp_path: Path):
        """不正なファイルで RuntimeError が発生する."""
        bad_path = tmp_path / "invalid.onnx"
        bad_path.write_bytes(b"invalid")
        resolver = InputShapeResolver()

        with pytest.raises(RuntimeError, match="入力形状を取得できません"):
            resolver.extract_static_shape(bad_path)

    def test_raises_runtime_error_for_missing_file(self, tmp_path: Path):
        """存在しないファイルで RuntimeError が発生する."""
        resolver = InputShapeResolver()

        with pytest.raises(RuntimeError):
            resolver.extract_static_shape(tmp_path / "nonexistent.onnx")
