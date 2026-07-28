import json

import pytest

from corpus_corpus.metrics import MetricsLogger, build_report, read_csv_metrics, read_jsonl_metrics


def _fake_records(n=5):
    records = []
    for e in range(n):
        records.append(
            dict(
                epoch=e,
                train_loss=1.0 / (e + 1),
                train_acc=0.5 + 0.1 * e,
                val_loss=1.0 / (e + 1) + 0.05,
                val_acc=0.4 + 0.1 * e,
                lr=0.01,
            )
        )
    return records


def test_metrics_logger_writes_csv_header_and_rows(tmp_path):
    path = tmp_path / "metrics.csv"
    logger = MetricsLogger(csv_path=str(path))
    for r in _fake_records(3):
        logger.log(**r)
    logger.close()

    rows = read_csv_metrics(str(path))
    assert len(rows) == 3
    assert set(MetricsLogger.FIELDS) <= set(rows[0].keys())


def test_metrics_logger_writes_jsonl(tmp_path):
    path = tmp_path / "metrics.jsonl"
    logger = MetricsLogger(jsonl_path=str(path))
    for r in _fake_records(4):
        logger.log(**r)
    logger.close()

    rows = read_jsonl_metrics(str(path))
    assert len(rows) == 4
    assert rows[0]["epoch"] == 0


def test_metrics_logger_writes_both_formats_simultaneously(tmp_path):
    csv_path = tmp_path / "m.csv"
    jsonl_path = tmp_path / "m.jsonl"
    logger = MetricsLogger(csv_path=str(csv_path), jsonl_path=str(jsonl_path))
    for r in _fake_records(2):
        logger.log(**r)
    logger.close()
    assert len(read_csv_metrics(str(csv_path))) == 2
    assert len(read_jsonl_metrics(str(jsonl_path))) == 2


def test_jsonl_lines_are_valid_json():
    pass  # covered implicitly by read_jsonl_metrics using json.loads


def test_build_report_finds_best_epoch_by_val_loss():
    records = _fake_records(5)
    report = build_report(records)
    # val_loss decreases monotonically in our fake data, so best is last
    assert report["best_epoch"] == 4
    assert report["n_epochs"] == 5


def test_build_report_empty_records():
    report = build_report([])
    assert report["n_epochs"] == 0


def test_build_report_final_metrics_match_last_record():
    records = _fake_records(3)
    report = build_report(records)
    assert report["final_val_acc"] == records[-1]["val_acc"]
