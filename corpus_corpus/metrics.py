"""Per-epoch metrics logging to CSV / JSONL, plus a run report summary."""

import csv
import json
import os


class MetricsLogger:
    """Appends one record per epoch to a CSV and/or a JSONL file."""

    FIELDS = [
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "lr",
    ]

    def __init__(self, csv_path=None, jsonl_path=None):
        self.csv_path = csv_path
        self.jsonl_path = jsonl_path
        self._csv_file = None
        self._csv_writer = None
        if self.csv_path:
            self._csv_file = open(self.csv_path, "w", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self.FIELDS)
            self._csv_writer.writeheader()
        if self.jsonl_path:
            # truncate/create
            open(self.jsonl_path, "w").close()

    def log(self, **record):
        row = {k: record.get(k) for k in self.FIELDS}
        if self._csv_writer:
            self._csv_writer.writerow(row)
            self._csv_file.flush()
        if self.jsonl_path:
            with open(self.jsonl_path, "a") as f:
                f.write(json.dumps(row) + "\n")

    def close(self):
        if self._csv_file:
            self._csv_file.close()


def read_jsonl_metrics(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def read_csv_metrics(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def build_report(records):
    """Summarize a completed run's per-epoch records.

    Expects each record to have numeric-ish string/float values for
    val_loss / val_acc / train_loss / train_acc / epoch.
    """
    if not records:
        return {"n_epochs": 0}

    def as_float(v):
        return float(v) if v is not None and v != "" else None

    val_losses = [as_float(r["val_loss"]) for r in records]
    val_accs = [as_float(r["val_acc"]) for r in records]

    best_idx = min(
        (i for i, v in enumerate(val_losses) if v is not None),
        key=lambda i: val_losses[i],
        default=None,
    )

    last = records[-1]
    report = {
        "n_epochs": len(records),
        "final_train_loss": as_float(last["train_loss"]),
        "final_train_acc": as_float(last["train_acc"]),
        "final_val_loss": as_float(last["val_loss"]),
        "final_val_acc": as_float(last["val_acc"]),
    }
    if best_idx is not None:
        best = records[best_idx]
        report["best_epoch"] = int(as_float(best["epoch"]))
        report["best_val_loss"] = val_losses[best_idx]
        report["best_val_acc"] = val_accs[best_idx]
    return report


def report_from_file(path):
    if path.endswith(".jsonl"):
        records = read_jsonl_metrics(path)
    else:
        records = read_csv_metrics(path)
    return build_report(records)
