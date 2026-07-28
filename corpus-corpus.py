#!/usr/bin/env python3
"""corpus-corpus: a minimal numpy-only training harness CLI.

Commands:
    corpus-corpus train --config config.json
    corpus-corpus eval  --checkpoint model.npz --config config.json
    corpus-corpus report --metrics metrics.csv

See README.md for the full write-up (architecture, toy task results,
gradient-check results, test coverage).
"""

import argparse
import sys

from corpus_corpus.checkpoint import load_checkpoint
from corpus_corpus.config import load_config
from corpus_corpus.metrics import report_from_file
from corpus_corpus.mlp import MLP
from corpus_corpus.toy_data import make_two_spirals, make_xor
from corpus_corpus.train import evaluate, train


def _load_toy_task(config):
    task = config.get("task", "xor")
    seed = config.get("seed", 0)
    if task == "xor":
        return make_xor(seed=seed)
    if task == "two_spirals":
        return make_two_spirals(seed=seed)
    raise ValueError(f"unknown toy task {task!r}")


def cmd_train(args):
    config = load_config(args.config)
    x, y = _load_toy_task(config)

    result = train(
        x,
        y,
        layer_sizes=config["layer_sizes"],
        activations=config.get("activations"),
        optimizer_name=config.get("optimizer", "adam"),
        lr=config.get("lr", 0.01),
        momentum=config.get("momentum", 0.9),
        batch_size=config.get("batch_size", 16),
        epochs=config.get("epochs", 200),
        val_fraction=config.get("val_fraction", 0.2),
        patience=config.get("patience", 20),
        warmup_steps=config.get("warmup_steps", 0),
        lr_schedule=config.get("lr_schedule", "cosine"),
        seed=config.get("seed", 0),
        checkpoint_path=config.get("checkpoint_path"),
        metrics_csv=config.get("metrics_csv"),
        metrics_jsonl=config.get("metrics_jsonl"),
        verbose=not args.quiet,
    )

    val_loss, val_acc = evaluate(result["mlp"], result["x_val"], result["y_val"])
    print(f"final val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
    print(f"best epoch={result['best_epoch']} best_val_loss={result['best_val_loss']:.4f}")
    if result["checkpoint_path"]:
        print(f"checkpoint saved to {result['checkpoint_path']}")
    return 0


def cmd_eval(args):
    config = load_config(args.config)
    x, y = _load_toy_task(config)
    mlp = MLP(config["layer_sizes"], activations=config.get("activations"), seed=config.get("seed", 0))
    load_checkpoint(args.checkpoint, mlp)
    loss, acc = evaluate(mlp, x, y)
    print(f"loss={loss:.4f} acc={acc:.4f}")
    return 0


def cmd_report(args):
    report = report_from_file(args.metrics)
    for k, v in report.items():
        print(f"{k}: {v}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog="corpus-corpus")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train a model from a config file")
    p_train.add_argument("--config", required=True)
    p_train.add_argument("--quiet", action="store_true")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval", help="evaluate a checkpoint")
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.add_argument("--config", required=True)
    p_eval.set_defaults(func=cmd_eval)

    p_report = sub.add_parser("report", help="summarize a metrics file")
    p_report.add_argument("--metrics", required=True)
    p_report.set_defaults(func=cmd_report)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
