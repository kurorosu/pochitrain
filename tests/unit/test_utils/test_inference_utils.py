"""inference_utils モジュールのテスト."""

import csv
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

import pochitrain.utils.inference_utils as inference_utils
from pochitrain.utils.inference_utils import (
    auto_detect_config_path,
    compute_confusion_matrix,
    get_default_output_base_dir,
    log_inference_result,
    post_process_logits,
    save_classification_report,
    save_confusion_matrix_image,
    validate_data_path,
    validate_model_path,
    write_inference_csv,
    write_inference_summary,
)


def _read_csv_rows(csv_path: Path) -> list[list[str]]:
    """CSVファイルを読み込み, 行配列として返す."""
    with open(csv_path, encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        return list(reader)


def _collect_info_messages(mock_logger: Mock) -> list[str]:
    """logger.info の呼び出しメッセージを配列で返す."""
    return [str(call.args[0]) for call in mock_logger.info.call_args_list if call.args]


class TestAutoDetectConfigPath:
    """auto_detect_config_path関数のテスト."""

    def test_basic_path(self, tmp_path):
        """基本的なパスからconfig.pyを検出."""
        model_path = tmp_path / "work_dirs" / "20260126_001" / "models" / "model.pth"
        result = auto_detect_config_path(model_path)
        expected = tmp_path / "work_dirs" / "20260126_001" / "config.py"
        assert result == expected

    def test_returns_path_object(self, tmp_path):
        """Path型を返すことを確認."""
        model_path = tmp_path / "a" / "b" / "model.pth"
        result = auto_detect_config_path(model_path)
        assert isinstance(result, Path)


class TestGetDefaultOutputBaseDir:
    """get_default_output_base_dir関数のテスト."""

    def test_basic_path(self, tmp_path):
        """基本的なパスからinference_resultsを返す."""
        model_path = tmp_path / "work_dirs" / "20260126_001" / "models" / "model.pth"
        result = get_default_output_base_dir(model_path)
        expected = tmp_path / "work_dirs" / "20260126_001" / "inference_results"
        assert result == expected

    def test_returns_path_object(self, tmp_path):
        """Path型を返すことを確認."""
        model_path = tmp_path / "a" / "b" / "model.onnx"
        result = get_default_output_base_dir(model_path)
        assert isinstance(result, Path)


class TestValidateModelPath:
    """validate_model_path関数のテスト."""

    @pytest.mark.parametrize(
        "exists",
        [True, False],
        ids=["existing-model-path", "missing-model-path"],
    )
    def test_validate_model_path(self, tmp_path, exists):
        """モデルパスの存在有無で分岐することを確認."""
        model_file = tmp_path / "model.pth"
        if exists:
            model_file.touch()
            result = validate_model_path(model_file)
            assert result is None
            return

        with pytest.raises(SystemExit):
            validate_model_path(model_file)


class TestValidateDataPath:
    """validate_data_path関数のテスト."""

    @pytest.mark.parametrize(
        "exists",
        [True, False],
        ids=["existing-data-path", "missing-data-path"],
    )
    def test_validate_data_path(self, tmp_path, exists):
        """データパスの存在有無で分岐することを確認."""
        data_dir = tmp_path / "data"
        if exists:
            data_dir.mkdir()
            result = validate_data_path(data_dir)
            assert result is None
            return

        with pytest.raises(SystemExit):
            validate_data_path(data_dir)


class TestWriteInferenceCsv:
    """write_inference_csv関数のテスト."""

    def test_basic_csv_output(self, tmp_path):
        """基本的なCSV出力."""
        output_dir = tmp_path / "output"
        image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
        predictions = [0, 1, 0]
        true_labels = [0, 1, 1]
        confidences = [0.95, 0.88, 0.72]
        class_names = ["cat", "dog"]

        csv_path = write_inference_csv(
            output_dir=output_dir,
            image_paths=image_paths,
            predictions=predictions,
            true_labels=true_labels,
            confidences=confidences,
            class_names=class_names,
        )

        assert csv_path.exists()
        assert csv_path.name == "inference_results.csv"

        rows = _read_csv_rows(csv_path)

        assert rows[0] == [
            "image_path",
            "predicted",
            "predicted_class",
            "true",
            "true_class",
            "confidence",
            "correct",
        ]
        assert len(rows) == 4  # ヘッダー + 3行

    def test_correct_flag(self, tmp_path):
        """correct列が正しく設定される."""
        output_dir = tmp_path / "output"
        csv_path = write_inference_csv(
            output_dir=output_dir,
            image_paths=["a.jpg", "b.jpg"],
            predictions=[0, 1],
            true_labels=[0, 0],
            confidences=[0.9, 0.8],
            class_names=["cat", "dog"],
        )

        rows = _read_csv_rows(csv_path)

        assert rows[1][6] == "True"
        assert rows[2][6] == "False"

    def test_custom_filename(self, tmp_path):
        """カスタムファイル名でCSV出力."""
        output_dir = tmp_path / "output"
        csv_path = write_inference_csv(
            output_dir=output_dir,
            image_paths=["a.jpg"],
            predictions=[0],
            true_labels=[0],
            confidences=[0.9],
            class_names=["cat"],
            filename="custom_results.csv",
        )
        assert csv_path.name == "custom_results.csv"

    def test_creates_output_dir(self, tmp_path):
        """出力ディレクトリが自動作成される."""
        output_dir = tmp_path / "nested" / "output"
        assert not output_dir.exists()

        write_inference_csv(
            output_dir=output_dir,
            image_paths=["a.jpg"],
            predictions=[0],
            true_labels=[0],
            confidences=[0.9],
            class_names=["cat"],
        )
        assert output_dir.exists()

    def test_returns_path(self, tmp_path):
        """戻り値がPath型."""
        result = write_inference_csv(
            output_dir=tmp_path,
            image_paths=["a.jpg"],
            predictions=[0],
            true_labels=[0],
            confidences=[0.9],
            class_names=["cat"],
        )
        assert isinstance(result, Path)

    def test_confidence_format(self, tmp_path):
        """信頼度が小数4桁でフォーマットされる."""
        csv_path = write_inference_csv(
            output_dir=tmp_path,
            image_paths=["a.jpg"],
            predictions=[0],
            true_labels=[0],
            confidences=[0.123456789],
            class_names=["cat"],
        )

        rows = _read_csv_rows(csv_path)

        assert rows[1][5] == "0.1235"


class TestWriteInferenceSummary:
    """write_inference_summary関数のテスト."""

    def test_basic_summary(self, tmp_path):
        """基本的なサマリー出力."""
        output_dir = tmp_path / "output"
        model_path = Path("models/model.pth")
        data_path = Path("data/val")

        summary_path = write_inference_summary(
            output_dir=output_dir,
            model_path=model_path,
            data_path=data_path,
            num_samples=100,
            accuracy=95.0,
            avg_time_per_image=2.5,
            total_samples=90,
            warmup_samples=10,
        )

        assert summary_path.exists()
        content = summary_path.read_text(encoding="utf-8")
        assert "95.00%" in content
        assert "2.50 ms/image" in content
        assert "100" in content

    def test_throughput_calculation(self, tmp_path):
        """スループットが正しく計算される."""
        summary_path = write_inference_summary(
            output_dir=tmp_path,
            model_path=Path("m.pth"),
            data_path=Path("d"),
            num_samples=10,
            accuracy=90.0,
            avg_time_per_image=5.0,  # 5ms -> 200 images/sec
            total_samples=10,
            warmup_samples=0,
        )

        content = summary_path.read_text(encoding="utf-8")
        assert "200.0 images/sec" in content

    def test_zero_avg_time(self, tmp_path):
        """平均推論時間が0の場合スループットが0になる."""
        summary_path = write_inference_summary(
            output_dir=tmp_path,
            model_path=Path("m.pth"),
            data_path=Path("d"),
            num_samples=10,
            accuracy=90.0,
            avg_time_per_image=0.0,
            total_samples=0,
            warmup_samples=0,
        )

        content = summary_path.read_text(encoding="utf-8")
        assert "0.0 images/sec" in content

    def test_extra_info(self, tmp_path):
        """追加情報が出力される."""
        summary_path = write_inference_summary(
            output_dir=tmp_path,
            model_path=Path("m.pth"),
            data_path=Path("d"),
            num_samples=10,
            accuracy=90.0,
            avg_time_per_image=1.0,
            total_samples=10,
            warmup_samples=0,
            extra_info={"実行プロバイダー": "CPUExecutionProvider"},
        )

        content = summary_path.read_text(encoding="utf-8")
        assert "実行プロバイダー: CPUExecutionProvider" in content

    def test_custom_filename(self, tmp_path):
        """カスタムファイル名でサマリー出力."""
        summary_path = write_inference_summary(
            output_dir=tmp_path,
            model_path=Path("m.pth"),
            data_path=Path("d"),
            num_samples=10,
            accuracy=90.0,
            avg_time_per_image=1.0,
            total_samples=10,
            warmup_samples=0,
            filename="custom_summary.txt",
        )
        assert summary_path.name == "custom_summary.txt"

    def test_creates_output_dir(self, tmp_path):
        """出力ディレクトリが自動作成される."""
        output_dir = tmp_path / "nested" / "output"
        assert not output_dir.exists()

        write_inference_summary(
            output_dir=output_dir,
            model_path=Path("m.pth"),
            data_path=Path("d"),
            num_samples=10,
            accuracy=90.0,
            avg_time_per_image=1.0,
            total_samples=10,
            warmup_samples=0,
        )
        assert output_dir.exists()


class TestLogInferenceResult:
    """log_inference_result関数のテスト."""

    def test_logs_core_metrics(self, monkeypatch):
        """基本メトリクスのログ内容を検証."""
        mock_logger = Mock()
        monkeypatch.setattr(inference_utils, "logger", mock_logger)

        log_inference_result(
            num_samples=100,
            correct=95,
            avg_time_per_image=2.5,
            total_samples=90,
            warmup_samples=10,
        )

        logged_messages = _collect_info_messages(mock_logger)
        assert mock_logger.info.call_count == 5
        assert any("推論画像枚数: 100枚" in message for message in logged_messages)
        assert any("精度: 95.00%" in message for message in logged_messages)
        assert any(
            "平均推論時間: 2.50 ms/image" in message for message in logged_messages
        )
        assert any(
            "スループット: 400.0 images/sec" in message for message in logged_messages
        )
        assert any(
            "計測詳細: 90枚, ウォームアップ除外: 10枚" in message
            for message in logged_messages
        )

    def test_logs_optional_input_size_and_total_time(self, monkeypatch):
        """入力サイズと全処理時間の追加ログ分岐を検証."""
        mock_logger = Mock()
        monkeypatch.setattr(inference_utils, "logger", mock_logger)

        log_inference_result(
            num_samples=10,
            correct=8,
            avg_time_per_image=2.0,
            total_samples=10,
            warmup_samples=0,
            avg_total_time_per_image=4.0,
            input_size=(3, 224, 224),
        )

        logged_messages = _collect_info_messages(mock_logger)
        assert mock_logger.info.call_count == 8
        assert any(
            "入力解像度: 224x224 (WxH), チャンネル数: 3" in message
            for message in logged_messages
        )
        assert any(
            "平均全処理時間: 4.00 ms/image" in message for message in logged_messages
        )
        assert any(
            "スループット: 250.0 images/sec, 計測範囲: 全処理" in message
            for message in logged_messages
        )

    @pytest.mark.parametrize(
        "kwargs, expected_message",
        [
            (
                {
                    "num_samples": 0,
                    "correct": 0,
                    "avg_time_per_image": 0.0,
                    "total_samples": 0,
                    "warmup_samples": 0,
                },
                "0.00%",
            ),
            (
                {
                    "num_samples": 10,
                    "correct": 5,
                    "avg_time_per_image": 0.0,
                    "total_samples": 10,
                    "warmup_samples": 0,
                },
                "0.0 images/sec",
            ),
        ],
        ids=["zero-samples", "zero-avg-time"],
    )
    def test_logs_zero_edge_cases(self, monkeypatch, kwargs, expected_message):
        """ゼロ系の境界ケースで期待メッセージが出ることを確認."""
        mock_logger = Mock()
        monkeypatch.setattr(inference_utils, "logger", mock_logger)

        log_inference_result(**kwargs)

        logged_messages = _collect_info_messages(mock_logger)
        assert any(expected_message in message for message in logged_messages)


class TestComputeConfusionMatrix:
    """compute_confusion_matrix関数のテスト."""

    def test_basic_case(self):
        """基本的なケース."""
        predicted = [0, 1, 2, 0, 1]
        true_labels = [0, 1, 2, 0, 2]
        cm = compute_confusion_matrix(predicted, true_labels, num_classes=3)

        assert cm.shape == (3, 3)
        assert cm[0, 0] == 2
        assert cm[1, 1] == 1
        assert cm[2, 2] == 1
        assert cm[2, 1] == 1

    def test_perfect_prediction(self):
        """完全正解の場合, 対角成分のみに値がある."""
        predicted = [0, 1, 2, 0, 1, 2]
        true_labels = [0, 1, 2, 0, 1, 2]
        cm = compute_confusion_matrix(predicted, true_labels, num_classes=3)

        for i in range(3):
            for j in range(3):
                if i == j:
                    assert cm[i, j] > 0
                else:
                    assert cm[i, j] == 0

    def test_all_wrong(self):
        """全て不正解の場合, 対角成分が0."""
        predicted = [1, 2, 0]
        true_labels = [0, 1, 2]
        cm = compute_confusion_matrix(predicted, true_labels, num_classes=3)

        for i in range(3):
            assert cm[i, i] == 0

    def test_empty_input(self):
        """空入力の場合, 全てゼロの行列."""
        cm = compute_confusion_matrix([], [], num_classes=3)
        assert cm.shape == (3, 3)
        assert np.all(cm == 0)

    def test_returns_numpy_array(self):
        """戻り値がnumpy配列."""
        cm = compute_confusion_matrix([0], [0], num_classes=2)
        assert isinstance(cm, np.ndarray)
        assert cm.dtype == np.int64


class TestSaveConfusionMatrixImage:
    """save_confusion_matrix_image関数のテスト."""

    def test_creates_image_file(self, tmp_path):
        """画像ファイルが生成される."""
        output_path = save_confusion_matrix_image(
            predicted_labels=[0, 1, 0],
            true_labels=[0, 1, 1],
            class_names=["cat", "dog"],
            output_dir=tmp_path,
        )

        assert output_path.exists()
        assert output_path.name == "confusion_matrix.png"
        assert output_path.stat().st_size > 0

    def test_custom_filename(self, tmp_path):
        """カスタムファイル名で画像を保存."""
        output_path = save_confusion_matrix_image(
            predicted_labels=[0, 1],
            true_labels=[0, 1],
            class_names=["a", "b"],
            output_dir=tmp_path,
            filename="custom_cm.png",
        )

        assert output_path.name == "custom_cm.png"

    def test_creates_output_dir(self, tmp_path):
        """出力ディレクトリが自動作成される."""
        output_dir = tmp_path / "nested" / "output"
        assert not output_dir.exists()

        save_confusion_matrix_image(
            predicted_labels=[0],
            true_labels=[0],
            class_names=["a"],
            output_dir=output_dir,
        )

        assert output_dir.exists()

    def test_custom_config(self, tmp_path):
        """カスタム設定で画像を保存."""
        cm_config = {
            "title": "Test Matrix",
            "cmap": "Reds",
            "figsize": (10, 8),
        }

        output_path = save_confusion_matrix_image(
            predicted_labels=[0, 1, 0],
            true_labels=[0, 1, 1],
            class_names=["cat", "dog"],
            output_dir=tmp_path,
            cm_config=cm_config,
        )

        assert output_path.exists()

    def test_returns_path(self, tmp_path):
        """戻り値がPath型."""
        result = save_confusion_matrix_image(
            predicted_labels=[0],
            true_labels=[0],
            class_names=["a"],
            output_dir=tmp_path,
        )
        assert isinstance(result, Path)

    def test_works_without_matplotlib_fontja(self, tmp_path, monkeypatch):
        """matplotlib_fontja 未導入でも画像保存できる."""
        original_import_module = inference_utils.importlib.import_module

        def fake_import_module(name):
            if name == "matplotlib_fontja":
                raise ModuleNotFoundError("No module named 'matplotlib_fontja'")
            return original_import_module(name)

        monkeypatch.setattr(
            inference_utils.importlib,
            "import_module",
            fake_import_module,
        )
        monkeypatch.setattr(
            inference_utils,
            "_MATPLOTLIB_FONTJA_WARNING_EMITTED",
            False,
        )

        output_path = save_confusion_matrix_image(
            predicted_labels=[0, 1, 0],
            true_labels=[0, 1, 1],
            class_names=["cat", "dog"],
            output_dir=tmp_path,
        )

        assert output_path.exists()


class TestSaveClassificationReport:
    """save_classification_report関数のテスト."""

    def test_basic_report(self, tmp_path):
        """基本的なレポート出力."""
        csv_path = save_classification_report(
            predicted_labels=[0, 1, 1, 0, 1],
            true_labels=[0, 1, 0, 0, 1],
            class_names=["cat", "dog"],
            output_dir=tmp_path,
        )

        assert csv_path.exists()
        assert csv_path.name == "classification_report.csv"

        reader = _read_csv_rows(csv_path)

        assert len(reader) == 5
        assert reader[0] == ["class", "precision", "recall", "f1-score", "support"]
        assert reader[1][0] == "cat"
        assert reader[2][0] == "dog"
        assert reader[3][0] == "macro avg"
        assert reader[4][0] == "weighted avg"

    def test_perfect_prediction(self, tmp_path):
        """全問正解の場合, precision/recall/f1がすべて1.0."""
        csv_path = save_classification_report(
            predicted_labels=[0, 1, 2],
            true_labels=[0, 1, 2],
            class_names=["a", "b", "c"],
            output_dir=tmp_path,
        )

        reader = _read_csv_rows(csv_path)

        for row in reader[1:4]:  # 各クラス行
            assert row[1] == "1.0000"  # precision
            assert row[2] == "1.0000"  # recall
            assert row[3] == "1.0000"  # f1-score

    def test_all_wrong(self, tmp_path):
        """全問不正解の場合, precision/recall/f1がすべて0."""
        csv_path = save_classification_report(
            predicted_labels=[1, 0],
            true_labels=[0, 1],
            class_names=["a", "b"],
            output_dir=tmp_path,
        )

        reader = _read_csv_rows(csv_path)

        for row in reader[1:3]:  # 各クラス行
            assert row[1] == "0.0000"  # precision
            assert row[2] == "0.0000"  # recall
            assert row[3] == "0.0000"  # f1-score

    def test_support_values(self, tmp_path):
        """supportが正解ラベルのサンプル数と一致."""
        csv_path = save_classification_report(
            predicted_labels=[0, 0, 1, 1, 1],
            true_labels=[0, 0, 0, 1, 1],
            class_names=["cat", "dog"],
            output_dir=tmp_path,
        )

        reader = _read_csv_rows(csv_path)

        assert reader[1][4] == "3"  # cat: 3サンプル
        assert reader[2][4] == "2"  # dog: 2サンプル
        assert reader[3][4] == "5"  # macro avg: total
        assert reader[4][4] == "5"  # weighted avg: total

    def test_custom_filename(self, tmp_path):
        """カスタムファイル名を指定."""
        csv_path = save_classification_report(
            predicted_labels=[0],
            true_labels=[0],
            class_names=["a"],
            output_dir=tmp_path,
            filename="custom_report.csv",
        )

        assert csv_path.name == "custom_report.csv"
        assert csv_path.exists()

    def test_creates_output_dir(self, tmp_path):
        """出力ディレクトリが存在しない場合に自動作成."""
        output_dir = tmp_path / "nested" / "dir"

        save_classification_report(
            predicted_labels=[0],
            true_labels=[0],
            class_names=["a"],
            output_dir=output_dir,
        )

        assert output_dir.exists()

    def test_returns_path(self, tmp_path):
        """戻り値がPath型."""
        result = save_classification_report(
            predicted_labels=[0],
            true_labels=[0],
            class_names=["a"],
            output_dir=tmp_path,
        )
        assert isinstance(result, Path)


class TestPostProcessLogits:
    """post_process_logits関数のテスト."""

    @pytest.mark.parametrize(
        "logits, expected_indices",
        [
            (np.array([[2.0, 1.0, 0.1], [0.5, 3.0, 0.2]]), [0, 1]),
            (np.array([[-1.0, -2.0, -0.5]]), [2]),
            (np.array([[5.0]]), [0]),
        ],
        ids=["basic-batch", "negative-logits", "single-class"],
    )
    def test_predicted_indices(self, logits, expected_indices):
        """logitの最大値に対応する予測インデックスを返す."""
        predicted, confidence = post_process_logits(logits)

        assert predicted.tolist() == expected_indices
        assert predicted.shape[0] == len(expected_indices)
        assert confidence.shape[0] == len(expected_indices)

    @pytest.mark.parametrize(
        "logits, expected",
        [
            (np.array([[1.0, 2.0, 3.0]]), "greater-than-uniform"),
            (np.array([[1.0, 1.0, 1.0]]), "uniform"),
            (np.array([[5.0]]), "single-class"),
        ],
        ids=["softmax-rank", "equal-logits", "single-class-confidence"],
    )
    def test_confidence_behavior(self, logits, expected):
        """softmax後の信頼度が入力条件ごとの期待を満たす."""
        _, confidence = post_process_logits(logits)
        value = confidence[0]

        assert 0.0 <= value <= 1.0
        if expected == "greater-than-uniform":
            assert value > (1.0 / 3.0)
        elif expected == "uniform":
            assert np.isclose(value, 1.0 / 3.0)
        else:
            assert np.isclose(value, 1.0)

    def test_large_logits_numerical_stability(self):
        """大きなlogit値でも数値的に安定."""
        logits = np.array([[1000.0, 999.0, 998.0]])
        predicted, confidence = post_process_logits(logits)

        assert predicted[0] == 0
        assert 0.0 <= confidence[0] <= 1.0
        assert not np.isnan(confidence[0])
        assert not np.isinf(confidence[0])

    def test_batch_processing(self):
        """バッチ処理が正しく動作."""
        logits = np.array(
            [
                [5.0, 1.0],
                [1.0, 5.0],
                [3.0, 3.0],
            ]
        )
        predicted, confidence = post_process_logits(logits)

        assert predicted.shape == (3,)
        assert predicted[0] == 0
        assert predicted[1] == 1
        assert predicted[2] in [0, 1]

    def test_returns_numpy_arrays(self):
        """戻り値がnumpy配列."""
        logits = np.array([[1.0, 2.0]])
        predicted, confidence = post_process_logits(logits)

        assert isinstance(predicted, np.ndarray)
        assert isinstance(confidence, np.ndarray)
