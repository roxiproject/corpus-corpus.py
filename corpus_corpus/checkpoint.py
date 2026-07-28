"""Checkpoint save/load for MLP weights, using plain .npz files."""

import numpy as np


def save_checkpoint(path, mlp, extra=None):
    """Save every layer's W/b into a single .npz file.

    Keys are named layer{i}_W / layer{i}_b so reload doesn't depend on
    object identity, only on the layer_sizes/activations used to rebuild
    the MLP shell before calling load_checkpoint.
    """
    arrays = {}
    for i, layer in enumerate(mlp.layers):
        arrays[f"layer{i}_W"] = layer.W
        arrays[f"layer{i}_b"] = layer.b
    arrays["n_layers"] = np.array(len(mlp.layers))
    if extra:
        for k, v in extra.items():
            arrays[f"extra_{k}"] = np.array(v)
    np.savez(path, **arrays)


def load_checkpoint(path, mlp):
    """Load weights from `path` into an existing MLP shell in place.

    The shell's layer_sizes/activations must match what was saved.
    """
    data = np.load(path)
    n_layers = int(data["n_layers"])
    if n_layers != len(mlp.layers):
        raise ValueError(
            f"checkpoint has {n_layers} layers but mlp has {len(mlp.layers)}"
        )
    for i, layer in enumerate(mlp.layers):
        layer.W = data[f"layer{i}_W"].copy()
        layer.b = data[f"layer{i}_b"].copy()
    return mlp


def load_extra(path):
    data = np.load(path)
    return {
        k[len("extra_") :]: data[k].item()
        for k in data.files
        if k.startswith("extra_")
    }
