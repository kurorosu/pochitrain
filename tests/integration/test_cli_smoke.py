"""CLIエントリポイントのスモーク統合テスト."""

import pytest

import pochitrain.cli.export_onnx as export_onnx_cli
import pochitrain.cli.pochi as pochi_cli


class TestCliSmoke:
    """CLIの軽量スモーク."""

    def test_pochi_help(self, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
        """`pochi` のヘルプが表示されることを確認."""
        monkeypatch.setattr("sys.argv", ["pochi", "--help"])
        with pytest.raises(SystemExit) as exc:
            pochi_cli.main()
        assert exc.value.code == 0

        captured = capsys.readouterr()
        assert "convert" in captured.out
        assert "train" in captured.out
        assert "infer" in captured.out

    def test_export_onnx_help(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys,
    ) -> None:
        """`export-onnx` のヘルプが表示されることを確認."""
        monkeypatch.setattr("sys.argv", ["export-onnx", "--help"])
        with pytest.raises(SystemExit) as exc:
            export_onnx_cli.main()
        assert exc.value.code == 0

        captured = capsys.readouterr()
        assert "--input-size" in captured.out
        assert "--num-classes" in captured.out
