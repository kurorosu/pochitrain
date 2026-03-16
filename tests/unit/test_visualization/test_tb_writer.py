"""TensorBoardWriter のユニットテスト."""

import logging

from pochitrain.visualization.tensorboard import TensorBoardWriter


class TestTensorBoardWriter:
    """TensorBoardWriter のテスト."""

    def test_record_epoch_creates_event_file(self, tmp_path):
        """record_epoch で TensorBoard イベントファイルが作成されること."""
        logger = logging.getLogger("test")
        writer = TensorBoardWriter(log_dir=tmp_path, logger=logger)

        writer.record_epoch(
            epoch=1,
            train_loss=0.5,
            train_accuracy=80.0,
            learning_rate=0.001,
        )
        writer.close()

        event_files = list(tmp_path.glob("events.out.tfevents.*"))
        assert len(event_files) == 1

    def test_record_epoch_with_validation_metrics(self, tmp_path):
        """検証メトリクス付きで record_epoch が正常に動作すること."""
        logger = logging.getLogger("test")
        writer = TensorBoardWriter(log_dir=tmp_path, logger=logger)

        writer.record_epoch(
            epoch=1,
            train_loss=0.5,
            train_accuracy=80.0,
            learning_rate=0.001,
            val_loss=0.3,
            val_accuracy=85.0,
        )
        writer.close()

        event_files = list(tmp_path.glob("events.out.tfevents.*"))
        assert len(event_files) == 1

    def test_record_epoch_with_layer_wise_rates(self, tmp_path):
        """層別学習率付きで record_epoch が正常に動作すること."""
        logger = logging.getLogger("test")
        writer = TensorBoardWriter(log_dir=tmp_path, logger=logger)

        writer.record_epoch(
            epoch=1,
            train_loss=0.5,
            train_accuracy=80.0,
            learning_rate=0.001,
            layer_wise_rates={"layer1": 0.0001, "layer2": 0.0005},
        )
        writer.close()

        event_files = list(tmp_path.glob("events.out.tfevents.*"))
        assert len(event_files) == 1

    def test_multiple_epochs(self, tmp_path):
        """複数エポックの記録が正常に動作すること."""
        logger = logging.getLogger("test")
        writer = TensorBoardWriter(log_dir=tmp_path, logger=logger)

        for epoch in range(1, 4):
            writer.record_epoch(
                epoch=epoch,
                train_loss=0.5 / epoch,
                train_accuracy=70.0 + epoch * 5,
                learning_rate=0.001 * (0.9**epoch),
                val_loss=0.4 / epoch,
                val_accuracy=75.0 + epoch * 5,
            )
        writer.close()

        event_files = list(tmp_path.glob("events.out.tfevents.*"))
        assert len(event_files) == 1

    def test_close_is_idempotent(self, tmp_path):
        """close を複数回呼んでもエラーにならないこと."""
        logger = logging.getLogger("test")
        writer = TensorBoardWriter(log_dir=tmp_path, logger=logger)

        writer.record_epoch(
            epoch=1,
            train_loss=0.5,
            train_accuracy=80.0,
            learning_rate=0.001,
        )
        writer.close()
        writer.close()
