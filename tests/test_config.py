import json

import pytest

from corpus_corpus.config import DEFAULT_CONFIG, load_config


def test_load_json_config_overrides_defaults(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"lr": 0.5, "epochs": 10}))
    config = load_config(str(path))
    assert config["lr"] == 0.5
    assert config["epochs"] == 10
    # unspecified keys fall back to defaults
    assert config["batch_size"] == DEFAULT_CONFIG["batch_size"]


def test_load_yaml_config_overrides_defaults(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("lr: 0.3\nepochs: 7\n")
    config = load_config(str(path))
    assert config["lr"] == 0.3
    assert config["epochs"] == 7


def test_load_config_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path/config.json")


def test_default_config_has_required_keys():
    for key in ["layer_sizes", "activations", "optimizer", "lr", "epochs"]:
        assert key in DEFAULT_CONFIG
