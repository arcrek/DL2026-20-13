"""Hold-out split leakage and integrity checks for Memotion 7k. Run: python tests/test_data.py"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import data  # noqa: E402


def test_no_text_ocr():
    root = os.path.join(os.path.dirname(__file__), "..")
    for d in ("src", "scripts"):
        for fn in os.listdir(os.path.join(root, d)):
            if fn == "preprocessing.py":
                continue
            fp = os.path.join(root, d, fn)
            if os.path.isfile(fp):
                with open(fp, encoding="utf-8") as f:
                    assert "text_ocr" not in f.read(), f"text_ocr referenced in {fp}"


def test_splits_and_weights():
    for s, n in data.EXPECTED_SIZES.items():
        assert len(data.load_jsonl(s)) == n, s
    rows = data.load_jsonl("train")
    counts = Counter(r["label"] for r in rows)
    assert [counts[c] for c in range(3)] == [518, 1762, 3313]
    w = data.class_weights(rows)
    assert w[0] > w[1] > w[2] > 0


def test_holdout_file():
    import json
    p = os.path.join(os.path.dirname(__file__), "..", "features", "train_holdout.json")
    d = json.load(open(p))
    assert not set(d["fit"]) & set(d["holdout"])
    assert len(d["class_weights"]) == 3


def test_holdout_no_leakage():
    main()


def main():
    rows = data.load_jsonl("train")
    fit, hold, grouping = data.make_holdout(rows, data_dir=data.DATA_DIR)
    by_id = {r["id"]: r for r in rows}

    assert not (set(fit) & set(hold)), "Overlap between fit and holdout set!"
    assert len(fit) + len(hold) == len(rows), "Row count mismatch in holdout split"

    fit_texts = {data._norm_text(by_id[i]["text"]) for i in fit}
    hold_texts = {data._norm_text(by_id[i]["text"]) for i in hold}
    shared_captions = fit_texts & hold_texts
    assert not shared_captions, f"{len(shared_captions)} captions shared between fit and hold-out!"

    for s in ["train", "validation", "test"]:
        s_rows = data.load_jsonl(s)
        labels = {r["label"] for r in s_rows}
        assert labels.issubset({0, 1, 2}), f"Unexpected labels in {s}: {labels}"

    hold_dist = Counter(by_id[i]["label"] for i in hold)
    print(f"PASS test_data: grouping={grouping}, fit={len(fit)}, holdout={len(hold)}, hold_dist={dict(sorted(hold_dist.items()))}")


if __name__ == "__main__":
    test_no_text_ocr()
    test_splits_and_weights()
    main()
    test_holdout_file()
