"""The real training loop: minibatching+shuffling, val split, early
stopping with patience, LR scheduling, checkpointing, and metrics logging.
"""

import numpy as np

from corpus_corpus.checkpoint import save_checkpoint
from corpus_corpus.data import iterate_minibatches, num_batches, train_val_split
from corpus_corpus.metrics import MetricsLogger
from corpus_corpus.mlp import MLP
from corpus_corpus.optim import build_optimizer
from corpus_corpus.schedule import warmup_then_cosine


def accuracy(logits_or_preds, y, already_preds=False):
    preds = logits_or_preds if already_preds else np.argmax(logits_or_preds, axis=1)
    return float(np.mean(preds == y))


def evaluate(mlp, x, y, batch_size=256):
    losses = []
    n_correct = 0
    from corpus_corpus.losses import softmax_cross_entropy

    for xb, yb in iterate_minibatches(x, y, batch_size, shuffle=False):
        logits = mlp.forward(xb)
        loss, _ = softmax_cross_entropy(logits, yb)
        losses.append(loss * len(yb))
        n_correct += int(np.sum(np.argmax(logits, axis=1) == yb))
    total = len(y)
    return sum(losses) / total, n_correct / total


class EarlyStopper:
    """Stops when val_loss hasn't improved by at least `min_delta` for
    `patience` consecutive epochs. Tracks the best epoch seen.
    """

    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.best_epoch = -1
        self.num_bad_epochs = 0
        self.should_stop = False

    def step(self, val_loss, epoch):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        if self.num_bad_epochs >= self.patience:
            self.should_stop = True
        return self.should_stop


def train(
    x,
    y,
    layer_sizes,
    activations=None,
    optimizer_name="adam",
    lr=0.01,
    momentum=0.9,
    batch_size=16,
    epochs=200,
    val_fraction=0.2,
    patience=20,
    warmup_steps=0,
    lr_schedule="cosine",
    seed=0,
    checkpoint_path=None,
    metrics_csv=None,
    metrics_jsonl=None,
    verbose=False,
):
    """Run a full training job. Returns a dict with the trained mlp,
    the best checkpoint path (if any), and the metrics records.
    """
    x_train, y_train, x_val, y_val = train_val_split(x, y, val_fraction, seed=seed)

    mlp = MLP(layer_sizes, activations=activations, seed=seed)
    flat_params = mlp.get_flat_params()
    param_arrays = [p for _, _, p in flat_params]
    if optimizer_name == "sgd":
        optimizer = build_optimizer(optimizer_name, param_arrays, lr=lr, momentum=momentum)
    else:
        optimizer = build_optimizer(optimizer_name, param_arrays, lr=lr)

    logger = MetricsLogger(csv_path=metrics_csv, jsonl_path=metrics_jsonl)
    stopper = EarlyStopper(patience=patience)

    steps_per_epoch = num_batches(len(x_train), batch_size)
    total_steps = steps_per_epoch * epochs
    shuffle_rng = np.random.default_rng(seed)

    records = []
    global_step = 0
    best_saved = False

    for epoch in range(epochs):
        epoch_losses = []
        n_correct = 0
        n_seen = 0
        for xb, yb in iterate_minibatches(
            x_train, y_train, batch_size, shuffle=True, rng=shuffle_rng
        ):
            current_lr = warmup_then_cosine(
                global_step, warmup_steps, total_steps, lr
            )
            optimizer.set_lr(current_lr)

            loss, logits, param_grads = mlp.loss_and_grads(xb, yb)
            grad_arrays = []
            for i, layer in enumerate(mlp.layers):
                grad_arrays.append(param_grads[i]["W"])
                grad_arrays.append(param_grads[i]["b"])
            optimizer.step(grad_arrays)

            epoch_losses.append(loss * len(yb))
            n_correct += int(np.sum(np.argmax(logits, axis=1) == yb))
            n_seen += len(yb)
            global_step += 1

        train_loss = sum(epoch_losses) / n_seen
        train_acc = n_correct / n_seen
        val_loss, val_acc = evaluate(mlp, x_val, y_val)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr,
        }
        records.append(record)
        logger.log(**record)
        if verbose:
            print(
                f"epoch {epoch:3d}  train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.4f}  val_loss={val_loss:.4f} "
                f"val_acc={val_acc:.4f}  lr={current_lr:.5f}"
            )

        improved = val_loss < stopper.best_loss - stopper.min_delta
        if improved and checkpoint_path:
            save_checkpoint(checkpoint_path, mlp, extra={"epoch": epoch})
            best_saved = True

        if stopper.step(val_loss, epoch):
            if verbose:
                print(f"early stopping at epoch {epoch} (best={stopper.best_epoch})")
            break

    logger.close()

    return {
        "mlp": mlp,
        "records": records,
        "best_epoch": stopper.best_epoch,
        "best_val_loss": stopper.best_loss,
        "checkpoint_path": checkpoint_path if best_saved else None,
        "x_val": x_val,
        "y_val": y_val,
    }
