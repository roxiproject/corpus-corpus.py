"""A small dense multi-layer perceptron with hand-derived backprop.

No autodiff engine here on purpose: every gradient below is written out
by hand (dense-layer local Jacobians composed via the chain rule) so the
math can be checked directly against numerical gradients in the test
suite. This is the classic "derive it yourself" style of a from-scratch
NN implementation.
"""

import numpy as np

from corpus_corpus.activations import ACTIVATIONS
from corpus_corpus.losses import softmax_cross_entropy


class Dense:
    """A single fully-connected layer: y = x @ W + b, then an activation."""

    def __init__(self, in_dim, out_dim, activation="relu", rng=None):
        rng = rng or np.random.default_rng()
        # He-style init, reasonable for relu/gelu.
        scale = np.sqrt(2.0 / in_dim)
        self.W = rng.normal(0.0, scale, size=(in_dim, out_dim)).astype(np.float64)
        self.b = np.zeros(out_dim, dtype=np.float64)
        if activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}")
        self.activation_name = activation
        self._act_fwd, self._act_bwd = ACTIVATIONS[activation]
        # caches populated during forward(), consumed during backward()
        self._x = None
        self._z = None

    def forward(self, x):
        self._x = x
        self._z = x @ self.W + self.b
        return self._act_fwd(self._z)

    def backward(self, grad_output):
        """Given dL/d(activation_output), return dL/d(x), dL/dW, dL/db."""
        grad_z = self._act_bwd(self._z, grad_output)
        grad_W = self._x.T @ grad_z
        grad_b = np.sum(grad_z, axis=0)
        grad_x = grad_z @ self.W.T
        return grad_x, grad_W, grad_b

    def params(self):
        return {"W": self.W, "b": self.b}

    def set_params(self, params):
        self.W = params["W"]
        self.b = params["b"]


class MLP:
    """A stack of Dense layers. The last layer must use activation='linear'
    (raw logits) so it can be paired with softmax cross-entropy.
    """

    def __init__(self, layer_sizes, activations=None, seed=0):
        if len(layer_sizes) < 2:
            raise ValueError("need at least an input and output size")
        n_layers = len(layer_sizes) - 1
        if activations is None:
            activations = ["relu"] * (n_layers - 1) + ["linear"]
        if len(activations) != n_layers:
            raise ValueError("activations must have one entry per layer")

        rng = np.random.default_rng(seed)
        self.layers = []
        for i in range(n_layers):
            self.layers.append(
                Dense(layer_sizes[i], layer_sizes[i + 1], activations[i], rng=rng)
            )

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad_output):
        """Backprop through every layer, returning per-layer param grads
        in the same order as self.layers.
        """
        grads = []
        grad = grad_output
        for layer in reversed(self.layers):
            grad, grad_W, grad_b = layer.backward(grad)
            grads.append({"W": grad_W, "b": grad_b})
        grads.reverse()
        return grads

    def loss_and_grads(self, x, labels):
        logits = self.forward(x)
        loss, grad_logits = softmax_cross_entropy(logits, labels)
        param_grads = self.backward(grad_logits)
        return loss, logits, param_grads

    def predict(self, x):
        logits = self.forward(x)
        return np.argmax(logits, axis=1)

    def get_flat_params(self):
        """Flat list of (layer_index, name, array) for optimizers/checkpoints."""
        out = []
        for i, layer in enumerate(self.layers):
            for name, arr in layer.params().items():
                out.append((i, name, arr))
        return out

    def num_params(self):
        return sum(p.size for _, _, p in self.get_flat_params())
