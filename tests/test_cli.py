"""CLI tests. The entry point file is corpus-corpus.py (hyphenated), so
it's loaded via importlib rather than a normal package import.
"""

import importlib.util
import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI_PATH = os.path.join(REPO_ROOT, "corpus-corpus.py")


def _load_cli_module():
    spec = importlib.util.spec_from_file_location("corpus_corpus_cli", CLI_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_module_loads():
    cli = _load_cli_module()
    assert hasattr(cli, "build_parser")


def test_cli_parses_train_command():
    cli = _load_cli_module()
    parser = cli.build_parser()
    args = parser.parse_args(["train", "--config", "config.json"])
    assert args.command == "train"
    assert args.config == "config.json"


def test_cli_parses_eval_command():
    cli = _load_cli_module()
    parser = cli.build_parser()
    args = parser.parse_args(["eval", "--checkpoint", "model.npz", "--config", "c.json"])
    assert args.command == "eval"
    assert args.checkpoint == "model.npz"


def test_cli_parses_report_command():
    cli = _load_cli_module()
    parser = cli.build_parser()
    args = parser.parse_args(["report", "--metrics", "metrics.csv"])
    assert args.command == "report"
    assert args.metrics == "metrics.csv"


def test_cli_requires_a_command():
    cli = _load_cli_module()
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_cli_train_requires_config_flag():
    cli = _load_cli_module()
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["train"])


def test_cli_train_end_to_end_subprocess(tmp_path):
    config = {
        "task": "xor",
        "layer_sizes": [2, 8, 2],
        "epochs": 20,
        "batch_size": 8,
        "lr": 0.1,
        "checkpoint_path": str(tmp_path / "model.npz"),
        "metrics_csv": str(tmp_path / "metrics.csv"),
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    result = subprocess.run(
        [sys.executable, CLI_PATH, "train", "--config", str(config_path), "--quiet"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "val_acc" in result.stdout
    assert os.path.exists(config["checkpoint_path"])
    assert os.path.exists(config["metrics_csv"])


def test_cli_eval_end_to_end_subprocess(tmp_path):
    config = {
        "task": "xor",
        "layer_sizes": [2, 8, 2],
        "epochs": 20,
        "batch_size": 8,
        "lr": 0.1,
        "checkpoint_path": str(tmp_path / "model.npz"),
        "metrics_csv": str(tmp_path / "metrics.csv"),
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    train_result = subprocess.run(
        [sys.executable, CLI_PATH, "train", "--config", str(config_path), "--quiet"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert train_result.returncode == 0

    eval_result = subprocess.run(
        [sys.executable, CLI_PATH, "eval", "--checkpoint", config["checkpoint_path"],
         "--config", str(config_path)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert eval_result.returncode == 0, eval_result.stderr
    assert "acc=" in eval_result.stdout


def test_cli_report_end_to_end_subprocess(tmp_path):
    config = {
        "task": "xor",
        "layer_sizes": [2, 8, 2],
        "epochs": 15,
        "batch_size": 8,
        "checkpoint_path": str(tmp_path / "model.npz"),
        "metrics_csv": str(tmp_path / "metrics.csv"),
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    subprocess.run(
        [sys.executable, CLI_PATH, "train", "--config", str(config_path), "--quiet"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    report_result = subprocess.run(
        [sys.executable, CLI_PATH, "report", "--metrics", config["metrics_csv"]],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert report_result.returncode == 0, report_result.stderr
    assert "n_epochs" in report_result.stdout
