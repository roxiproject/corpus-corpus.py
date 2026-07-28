"""Config loading for training runs.

Supports JSON always. If PyYAML is importable, .yaml/.yml configs are
also supported; otherwise those raise a clear error telling the caller
to use JSON instead. We do not vendor a hand-rolled YAML parser since
pyyaml was already available in this environment.
"""

import json

try:
    import yaml

    _HAVE_YAML = True
except ImportError:  # pragma: no cover - environment dependent
    _HAVE_YAML = False

DEFAULT_CONFIG = {
    "layer_sizes": [2, 16, 16, 2],
    "activations": None,  # None => MLP auto-generates relu/.../linear
    "optimizer": "adam",
    "lr": 0.01,
    "momentum": 0.9,
    "batch_size": 16,
    "epochs": 200,
    "val_fraction": 0.2,
    "patience": 20,
    "warmup_steps": 0,
    "lr_schedule": "cosine",
    "seed": 0,
    "task": "xor",
    "checkpoint_path": "model.npz",
    "metrics_csv": "metrics.csv",
    "metrics_jsonl": None,
}


def load_config(path):
    if path.endswith((".yaml", ".yml")):
        if not _HAVE_YAML:
            raise RuntimeError(
                "PyYAML is not installed; use a .json config instead"
            )
        with open(path) as f:
            user_config = yaml.safe_load(f) or {}
    else:
        with open(path) as f:
            user_config = json.load(f)

    config = dict(DEFAULT_CONFIG)
    config.update(user_config)
    return config
