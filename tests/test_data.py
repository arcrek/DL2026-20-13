"""Hold-out split leakage and integrity checks for Memotion 7k. Run: python tests/test_data.py"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import data  # noqa: E402


def main():
    rows = data.load_jsonl("train")
    fit, hold, grouping = data.make_holdout(rows)
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
    main()
