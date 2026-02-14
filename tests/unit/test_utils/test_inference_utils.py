"""inference_utils モジュールのテスト."""

import csv
import tempfile
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

    def test_existing_model_path(self, tmp_path):
        """存在するモデルパスではSystemExitが発生しない."""
        model_file = tmp_path / "model.pth"
        model_file.touch()
        result = validate_model_path(model_file)
        assert result is None

    def test_non_existing_model_path(self, tmp_path):
        """存在しないモデルパスでSystemExitが発生する."""
        model_file = tmp_path / "non_existent.pth"
        with pytest.raises(SystemExit):
            validate_model_path(model_file)


class TestValidateDataPath:
    """validate_data_path関数のテスト."""

    def test_existing_data_path(self, tmp_path):
        """存在するデータパスではSystemExitが発生しない."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        result = validate_data_path(data_dir)
        assert result is None

    def test_non_existing_data_path(self, tmp_path):
        """存在しないデータパスでSystemExitが発生する."""
        data_dir = tmp_path / "non_existent"
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

        # CSV内容の検証
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

        # ヘッダー行
        assert rows[0] == [
            "image_path",
            "predicted",
            "predicted_class",
            "true",
            "true_class",
            "confidence",
            "correct",
        ]
        # データ行数
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

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

        # 1行目: pred=0, true=0 -> correct=True
        assert rows[1][6] == "True"
        # 2行目: pred=1, true=0 -> correct=False
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

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

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

        logged_messages = [
            str(call.args[0]) for call in mock_logger.info.call_args_list if call.args
        ]
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

        logged_messages = [
            str(call.args[0]) for call in mock_logger.info.call_args_list if call.args
        ]
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

    def test_zero_samples(self, monkeypatch):
        mock_logger = Mock()
        monkeypatch.setattr(inference_utils, "logger", mock_logger)

        log_inference_result(
            num_samples=0,
            correct=0,
            avg_time_per_image=0.0,
            total_samples=0,
            warmup_samples=0,
        )

        logged_messages = [
            str(call.args[0]) for call in mock_logger.info.call_args_list if call.args
        ]
        assert any("0.00%" in message for message in logged_messages)

    def test_zero_avg_time(self, monkeypatch):
        mock_logger = Mock()
        monkeypatch.setattr(inference_utils, "logger", mock_logger)

        log_inference_result(
            num_samples=10,
            correct=5,
            avg_time_per_image=0.0,
            total_samples=10,
            warmup_samples=0,
        )

        logged_messages = [
            str(call.args[0]) for call in mock_logger.info.call_args_list if call.args
        ]
        assert any("0.0 images/sec" in message for message in logged_messages)


class TestComputeConfusionMatrix:
    """compute_confusion_matrix関数のテスト."""

    def test_basic_case(self):
        """基本的なケース."""
        predicted = [0, 1, 2, 0, 1]
        true_labels = [0, 1, 2, 0, 2]
        cm = compute_confusion_matrix(predicted, true_labels, num_classes=3)

        assert cm.shape == (3, 3)
        # 正解: (0,0)=2, (1,1)=1, (2,2)=1, 誤り: (2,1)=1
        assert cm[0, 0] == 2
        assert cm[1, 1] == 1
        assert cm[2, 2] == 1
        assert cm[2, 1] == 1

    def test_perfect_prediction(self):
        """完全正解の場合, 対角成分のみに値がある."""
        predicted = [0, 1, 2, 0, 1, 2]
        true_labels = [0, 1, 2, 0, 1, 2]
        cm = compute_confusion_matrix(predicted, true_labels, num_classes=3)

        # 対角成分のみ非ゼロ
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

        with open(csv_path, encoding="utf-8") as f:
            reader = list(csv.reader(f))

        # ヘッダー + クラス数(2) + macro avg + weighted avg = 5行
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

        with open(csv_path, encoding="utf-8") as f:
            reader = list(csv.reader(f))

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

        with open(csv_path, encoding="utf-8") as f:
            reader = list(csv.reader(f))

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

        with open(csv_path, encoding="utf-8") as f:
            reader = list(csv.reader(f))

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

    def test_basic_case(self):
        """基本的なケース: 正しいshapeと値の範囲."""
        logits = np.array([[2.0, 1.0, 0.1], [0.5, 3.0, 0.2]])
        predicted, confidence = post_process_logits(logits)

        assert predicted.shape == (2,)
        assert confidence.shape == (2,)
        assert predicted[0] == 0  # 最大値はindex 0
        assert predicted[1] == 1  # 最大値はindex 1

    def test_confidence_range(self):
        """信頼度が0-1の範囲内."""
        logits = np.array([[1.0, 2.0, 3.0]])
        predicted, confidence = post_process_logits(logits)

        assert 0.0 <= confidence[0] <= 1.0

    def test_softmax_probabilities_sum_to_one(self):
        """softmax適用後の確率の合計が1になることを間接的に確認."""
        logits = np.array([[1.0, 2.0, 3.0]])
        predicted, confidence = post_process_logits(logits)

        # 3クラスで最大logitが3.0の場合, confidence > 1/3
        assert confidence[0] > 1.0 / 3.0

    def test_single_class(self):
        """1クラスの場合, confidenceが1.0."""
        logits = np.array([[5.0]])
        predicted, confidence = post_process_logits(logits)

        assert predicted[0] == 0
        assert np.isclose(confidence[0], 1.0)

    def test_equal_logits(self):
        """全て同じlogitの場合, confidenceが均等."""
        logits = np.array([[1.0, 1.0, 1.0]])
        predicted, confidence = post_process_logits(logits)

        assert np.isclose(confidence[0], 1.0 / 3.0)

    def test_large_logits_numerical_stability(self):
        """大きなlogit値でも数値的に安定."""
        logits = np.array([[1000.0, 999.0, 998.0]])
        predicted, confidence = post_process_logits(logits)

        assert predicted[0] == 0
        assert 0.0 <= confidence[0] <= 1.0
        assert not np.isnan(confidence[0])
        assert not np.isinf(confidence[0])

    def test_negative_logits(self):
        """負のlogitでも正しく動作."""
        logits = np.array([[-1.0, -2.0, -0.5]])
        predicted, confidence = post_process_logits(logits)

        assert predicted[0] == 2  # -0.5が最大
        assert 0.0 <= confidence[0] <= 1.0

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
        # 3番目は同値なのでどちらでもOK
        assert predicted[2] in [0, 1]

    def test_returns_numpy_arrays(self):
        """戻り値がnumpy配列."""
        logits = np.array([[1.0, 2.0]])
        predicted, confidence = post_process_logits(logits)

        assert isinstance(predicted, np.ndarray)
        assert isinstance(confidence, np.ndarray)
