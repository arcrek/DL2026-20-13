"""Memotion 7k (SemEval-2020 Task 8) data loader, hold-out split, and verification."""
import hashlib
import json
import os
from collections import Counter

import numpy as np

DATA_DIR = os.environ.get("DATA_DIR", "data/memotion")
FEATURE_DIR = os.environ.get("FEATURE_DIR", "features")
SPLITS = ["train", "validation", "test"]
HOLDOUT_FRAC = 0.10
HOLDOUT_SEED = 0
EXPECTED_SIZES = {"train": 5593, "validation": 699, "test": 700}
NUM_CLASSES = 3
CLASS_NAMES = ["negative", "neutral", "positive"]


def _norm_text(t):
    return " ".join((t or "").lower().split())


def load_jsonl(split, data_dir=DATA_DIR):
    path = os.path.join(data_dir, f"{split}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing split file: {path}")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                r["id"] = str(r["id"])
                rows.append(r)
    return rows


def group_ids(rows, data_dir=None):
    """Group memes that share a caption or identical image using Union-Find."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    hashed = 0
    for r in rows:
        keys = [("t", _norm_text(r["text"]))]
        path = os.path.join(data_dir, r["img"]) if data_dir else None
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                keys.append(("i", hashlib.md5(f.read()).hexdigest()))
            hashed += 1
        for k in keys:
            parent[find(k)] = find(("m", r["id"]))

    grouping_type = "text+image" if (hashed == len(rows) and hashed > 0) else "text"
    return {r["id"]: find(("m", r["id"])) for r in rows}, grouping_type


def make_holdout(rows, frac=HOLDOUT_FRAC, seed=HOLDOUT_SEED, data_dir=None):
    """Group-aware deterministic split of train into (fit_ids, holdout_ids)."""
    gid, grouping = group_ids(rows, data_dir)
    groups = {}
    for r in rows:
        groups.setdefault(gid[r["id"]], []).append(r["id"])
    keys = sorted(groups.keys(), key=lambda k: min(groups[k]))
    np.random.RandomState(seed).shuffle(keys)
    target = int(round(len(rows) * frac))
    hold = []
    for k in keys:
        if len(hold) >= target:
            break
        hold += groups[k]
    hs = set(hold)
    return sorted(i for i in gid if i not in hs), sorted(hold), grouping


def class_weights(rows, num_classes=NUM_CLASSES):
    """Inverse-frequency weights, normalized so that they average to 1 over samples."""
    counts = Counter(r["label"] for r in rows)
    n = len(rows)
    return [n / (num_classes * counts[c]) for c in range(num_classes)]


def save_holdout(data_dir=DATA_DIR, out=None):
    out = out or os.path.join(os.environ.get("FEATURE_DIR", "features"), "train_holdout.json")
    rows = load_jsonl("train", data_dir)
    fit, hold, grouping = make_holdout(rows, data_dir=data_dir)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "seed": HOLDOUT_SEED,
            "frac": HOLDOUT_FRAC,
            "grouping": grouping,
            "class_weights": class_weights([r for r in rows if r["id"] in set(fit)]),
            "fit": fit,
            "holdout": hold
        }, f, indent=1)
    return fit, hold


def verify(data_dir=DATA_DIR, check_images=False):
    """Verify split sizes, label validity, image existence, and id uniqueness."""
    ok = True
    data = {}
    for s in SPLITS:
        rows = load_jsonl(s, data_dir)
        data[s] = rows
        counts = Counter(r["label"] for r in rows)
        exp = EXPECTED_SIZES.get(s)
        match = (exp == len(rows))
        ok = ok and match
        print(f"{s:11s} n={len(rows):5d} labels={dict(sorted(counts.items()))} {'OK' if match else 'FAIL'}")
        assert len({r['id'] for r in rows}) == len(rows), f"duplicate IDs in {s}"
        for r in rows:
            assert r["label"] in (0, 1, 2), f"invalid label {r['label']} in {s}"
        if check_images:
            missing = [r["img"] for r in rows if not os.path.exists(os.path.join(data_dir, r["img"]))]
            assert not missing, f"{len(missing)} missing images in {s}, e.g. {missing[:3]}"
            print(f"  All {len(rows)} images verified for {s}.")

    # Check cross-split ID overlaps
    ids = {s: {r['id'] for r in rows} for s, rows in data.items()}
    for a in ids:
        for b in ids:
            if a < b and ids[a] & ids[b]:
                print(f"Overlap {a}/{b}: {len(ids[a] & ids[b])}")
    return ok


if __name__ == "__main__":
    import sys
    check_img = "--images" in sys.argv
    is_ok = verify(check_images=check_img)
    fit_set, hold_set = save_holdout()
    train_rows = {r["id"]: r["label"] for r in load_jsonl("train")}
    hold_dist = Counter(train_rows[i] for i in hold_set)
    print(f"Holdout: fit={len(fit_set)} holdout={len(hold_set)} dist={dict(sorted(hold_dist.items()))}")
    sys.exit(0 if is_ok else 1)
