"""Smoke test for src/heads.py on synthetic 3-class multimodal features. Run: python tests/test_heads.py"""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import heads  # noqa: E402

D = 32
NUM_CLASSES = 3


def make_synthetic(n, rng, w_i, w_t, name):
    i = rng.randn(n, D)
    t = rng.randn(n, D)
    logits = i @ w_i + t @ w_t
    y = np.argmax(logits, axis=-1)
    ids = np.array([f"{name}_{k:05d}" for k in range(n)])
    return i, t, y, ids


def main():
    rng = np.random.RandomState(42)
    w_i = rng.randn(D, NUM_CLASSES)
    w_t = rng.randn(D, NUM_CLASSES)
    tmp = tempfile.mkdtemp()

    sizes = {"train": 1000, "validation": 200, "test": 200}
    for s, n in sizes.items():
        i, t, y, ids = make_synthetic(n, rng, w_i, w_t, s)
        for k, v in [("img", i), ("txt", t), ("labels", y), ("ids", ids)]:
            np.save(os.path.join(tmp, f"{s}_{k}.npy"), v)

    train_ids = [str(x) for x in np.load(os.path.join(tmp, "train_ids.npy"))]
    hold_ids = train_ids[::10]
    fit_ids = [x for x in train_ids if x not in set(hold_ids)]
    with open(os.path.join(tmp, "train_holdout.json"), "w", encoding="utf-8") as f:
        json.dump({"fit": fit_ids, "holdout": hold_ids}, f)

    data = heads.load_features(tmp)
    assert len(data["fit"]["y"]) + len(data["holdout"]["y"]) == sizes["train"]
    assert not (set(data["fit"]["ids"]) & set(data["holdout"]["ids"]))

    heads.MAX_EPOCHS = 10
    heads.PATIENCE = 5
    heads.RESULTS_DIR = os.path.join(tmp, "results")
    heads.SEEDS = [0, 1]
    heads.HEADS = ["img", "txt", "concat", "gated"]

    cfg = {"lr": 3e-3, "wd": 1e-4, "dropout": 0.1}
    cfgs = {k: cfg for k in heads.HEADS}

    print("Running synthetic heads smoke test...")
    heads.run(data, "cpu", cfgs)
    res_summary = heads.summary(heads.RESULTS_DIR)

    for k in heads.HEADS:
        assert k in res_summary, f"Missing {k} in summary"
        f1 = res_summary[k]["validation_macro_f1"]["mean"]
        acc = res_summary[k]["validation_acc"]["mean"]
        print(f"PASS head {k:10s} val_macro_f1={f1:.3f} val_acc={acc:.3f}")

    print("ALL TESTS PASSED for test_heads.py")


if __name__ == "__main__":
    main()
