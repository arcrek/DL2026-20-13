"""Schema and config contract tests. Run: python tests/test_results.py (or pytest tests/)"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import config  # noqa: E402
import results  # noqa: E402


def _fake(n=20, seed=0):
    rng = np.random.RandomState(seed)
    p = rng.dirichlet(np.ones(3), n)
    return dict(ids=[f"test_{i:05d}" for i in range(n)], y_true=rng.randint(0, 3, n), probs=p,
                sarcasm=["general"] * n)


def test_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = results.save_results("text", 1, results_dir=d, **_fake())
        assert os.path.basename(path) == "text_seed1.json"
        r = results.load_results("text", 1, results_dir=d)
        assert r["y_pred"] == np.argmax(r["probs"], 1).tolist()
        assert 0 <= r["accuracy"] <= 1 and 0 <= r["macro_f1"] <= 1


def test_rejects_bad_probs_and_lengths():
    f = _fake()
    with tempfile.TemporaryDirectory() as d:
        bad = dict(f, probs=np.asarray(f["probs"]) * 2)
        for kw in (bad, dict(f, sarcasm=f["sarcasm"][:-1]), dict(f, ids=f["ids"][:-1] + f["ids"][:1])):
            try:
                results.save_results("text", 0, results_dir=d, **kw)
            except ValueError:
                continue
            raise AssertionError("invalid result accepted")


def test_config():
    cfg = config.load_config()
    assert cfg["seeds"] == [0, 1, 2]
    assert "text" in cfg["configs"] and "image" in cfg["configs"]


if __name__ == "__main__":
    test_roundtrip()
    test_rejects_bad_probs_and_lengths()
    test_config()
    print("OK")
