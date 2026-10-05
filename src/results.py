"""Shared results contract. Every model branch (text, image, both_*) writes through `save_results()`.

File: results/{config}_seed{seed}.json  (e.g. results/text_seed0.json, results/both_concat_seed2.json)

Schema (version 1), all lists aligned to `ids` (test split order):
  schema_version int   = 1
  config         str   e.g. "text", "image", "both_concat", "both_cross_attn", "both_product"
  seed           int   0, 1 or 2
  split          str   evaluated split, "test"
  ids            [str] sample ids
  y_true         [int] gold sentiment, 0=negative 1=neutral 2=positive
  y_pred         [int] argmax of probs
  probs          [[float]*3] softmax probabilities, rows sum to 1
  sarcasm        [str] raw sarcasm label per sample (for sub-group analysis)
  accuracy       float computed here from y_true/y_pred
  macro_f1       float computed here from y_true/y_pred
  extra          dict  optional free-form (hyperparameters, best epoch, ...)
"""
import json
import os

import numpy as np

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")
SCHEMA_VERSION = 1
NUM_CLASSES = 3
REQUIRED_KEYS = ("schema_version", "config", "seed", "split", "ids", "y_true", "y_pred",
                 "probs", "sarcasm", "accuracy", "macro_f1")


def result_path(config, seed, results_dir=None):
    return os.path.join(results_dir or RESULTS_DIR, f"{config}_seed{seed}.json")


def validate_result(res):
    """Raise ValueError if `res` violates the schema; return it otherwise."""
    missing = [k for k in REQUIRED_KEYS if k not in res]
    if missing:
        raise ValueError(f"missing keys: {missing}")
    if res["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {res['schema_version']}")
    n = len(res["ids"])
    for k in ("y_true", "y_pred", "probs", "sarcasm"):
        if len(res[k]) != n:
            raise ValueError(f"'{k}' length {len(res[k])} != len(ids) {n}")
    if len(set(res["ids"])) != n:
        raise ValueError("duplicate ids")
    probs = np.asarray(res["probs"], dtype=float)
    if probs.shape != (n, NUM_CLASSES) or not np.isfinite(probs).all():
        raise ValueError(f"probs must be finite with shape ({n}, {NUM_CLASSES})")
    if not np.allclose(probs.sum(1), 1.0, atol=1e-3):
        raise ValueError("probs rows must sum to 1")
    for k in ("y_true", "y_pred"):
        a = np.asarray(res[k])
        if a.size and (a.min() < 0 or a.max() >= NUM_CLASSES):
            raise ValueError(f"'{k}' labels out of range 0..{NUM_CLASSES - 1}")
    return res


def _macro_f1(y_true, y_pred):
    f1s = []
    for c in range(NUM_CLASSES):
        tp = float(((y_pred == c) & (y_true == c)).sum())
        fp = float(((y_pred == c) & (y_true != c)).sum())
        fn = float(((y_pred != c) & (y_true == c)).sum())
        d = 2 * tp + fp + fn
        f1s.append(2 * tp / d if d else 0.0)
    return float(np.mean(f1s))


def save_results(config, seed, ids, y_true, probs, sarcasm, split="test", extra=None,
                 results_dir=None):
    """Validate and write one (config, seed) result file. Returns the path.

    y_pred, accuracy and macro_f1 are derived here so every branch reports identically.
    """

    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true).astype(int)
    y_pred = probs.argmax(1)
    res = {
        "schema_version": SCHEMA_VERSION,
        "config": str(config),
        "seed": int(seed),
        "split": split,
        "ids": [str(i) for i in ids],
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "probs": probs.tolist(),
        "sarcasm": [str(s) for s in sarcasm],
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": _macro_f1(y_true, y_pred),
        "extra": extra or {},
    }
    validate_result(res)
    path = result_path(config, seed, results_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f)
    return path


def load_results(config, seed, results_dir=None):
    with open(result_path(config, seed, results_dir), encoding="utf-8") as f:
        return validate_result(json.load(f))
