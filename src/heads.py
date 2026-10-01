"""Linear probes and fusion heads over cached frozen CLIP ViT-L/14 features for Memotion 7k.

Usage:
  python src/heads.py --tune     # grid search on train hold-out -> configs/heads.json
  python src/heads.py --run      # (auto-tunes if missing/stale) seeds 0-4 -> results/heads/*.json
  python src/heads.py --summary  # aggregate mean +- std -> results/heads_summary.json

Evaluation metric: Macro F1-score & Accuracy (SemEval-2020 Task 8 standard).
Heads: img, txt (linear probes), concat_linear (linear baseline), concat, product, gated (fusion).
"""
import argparse
import copy
import hashlib
import itertools
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

FEATURE_DIR = os.environ.get("FEATURE_DIR", "features")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")
CONFIG_PATH = "configs/heads.json"
HEADS = ["img", "txt", "concat_linear", "concat", "product", "gated"]
SEEDS = [0, 1, 2, 3, 4]
EVAL_SPLITS = ["validation", "test"]
NUM_CLASSES = 3
GRID = {"lr": [1e-3, 3e-4], "wd": [1e-4, 1e-2], "dropout": [0.1, 0.4]}
HIDDEN = 256
MAX_EPOCHS, PATIENCE, BATCH = 80, 12, 128


def _norm(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)


def load_features(feature_dir=FEATURE_DIR):
    """Load features and carve 'fit' and 'holdout' from train using features/train_holdout.json."""
    raw = {}
    for s in ["train"] + EVAL_SPLITS:
        raw[s] = {
            "img": _norm(np.load(os.path.join(feature_dir, f"{s}_img.npy"))),
            "txt": _norm(np.load(os.path.join(feature_dir, f"{s}_txt.npy"))),
            "y": np.load(os.path.join(feature_dir, f"{s}_labels.npy")),
            "ids": np.load(os.path.join(feature_dir, f"{s}_ids.npy"))
        }

    ho_path = os.path.join(feature_dir, "train_holdout.json")
    with open(ho_path, encoding="utf-8") as f:
        ho = json.load(f)

    raw["holdout_id"] = hashlib.sha1("\n".join(sorted(ho["holdout"])).encode()).hexdigest()[:12]
    pos = {ident: idx for idx, ident in enumerate(raw["train"]["ids"])}
    for name in ("fit", "holdout"):
        indices = np.array([pos[i] for i in ho[name]])
        raw[name] = {k: v[indices] for k, v in raw["train"].items()}

    return raw


class Head(nn.Module):
    def __init__(self, kind, d, dropout, num_classes=NUM_CLASSES):
        super().__init__()
        self.kind = kind
        mlp = lambda in_dim: nn.Sequential(
            nn.Linear(in_dim, HIDDEN),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(HIDDEN, num_classes)
        )
        if kind in ("img", "txt"):
            self.net = nn.Linear(d, num_classes)
        elif kind == "concat_linear":
            self.net = nn.Linear(2 * d, num_classes)
        elif kind == "concat":
            self.net = mlp(2 * d)
        elif kind == "product":
            self.scale = float(d)
            self.net = mlp(d)
        elif kind == "gated":
            self.pi = nn.Linear(d, HIDDEN)
            self.pt = nn.Linear(d, HIDDEN)
            self.gate = nn.Linear(2 * d, HIDDEN)
            self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(HIDDEN, num_classes))
        else:
            raise ValueError(f"Unknown head kind: {kind}")

    def forward(self, i, t):
        k = self.kind
        if k == "img":
            return self.net(i)
        if k == "txt":
            return self.net(t)
        if k in ("concat_linear", "concat"):
            return self.net(torch.cat([i, t], -1))
        if k == "product":
            return self.net(i * t * self.scale)
        g = torch.sigmoid(self.gate(torch.cat([i, t], -1)))
        fused = g * torch.tanh(self.pi(i)) + (1.0 - g) * torch.tanh(self.pt(t))
        return self.out(fused)


def _t(d, dev):
    return (
        torch.tensor(d["img"], dtype=torch.float32, device=dev),
        torch.tensor(d["txt"], dtype=torch.float32, device=dev),
        torch.tensor(d["y"], dtype=torch.long, device=dev)
    )


@torch.no_grad()
def predict_proba(model, d, dev):
    model.eval()
    i, t, _ = _t(d, dev)
    logits = model(i, t)
    return torch.softmax(logits, dim=-1).cpu().numpy()


def compute_metrics(y_true, probas):
    preds = np.argmax(probas, axis=-1)
    return {
        "macro_f1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, preds, average="weighted", zero_division=0)),
        "acc": float(accuracy_score(y_true, preds))
    }


def train_head(kind, data, cfg, seed, dev="cpu"):
    """Train on 'fit', early-stop on 'holdout' Macro F1."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    d = data["fit"]["img"].shape[1]
    model = Head(kind, d, cfg["dropout"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    loss_fn = nn.CrossEntropyLoss()

    i, t, y = _t(data["fit"], dev)
    g = torch.Generator().manual_seed(seed)
    best_model, best_f1, best_ep, patience_cnt = None, -1.0, 0, 0

    for ep in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(len(y), generator=g).to(dev)
        for b in range(0, len(y), BATCH):
            idx = perm[b:b + BATCH]
            logits = model(i[idx], t[idx])
            loss = loss_fn(logits, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

        val_probs = predict_proba(model, data["holdout"], dev)
        f1 = compute_metrics(data["holdout"]["y"], val_probs)["macro_f1"]
        if f1 > best_f1:
            best_model, best_f1, best_ep, patience_cnt = copy.deepcopy(model), f1, ep, 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                break

    return best_model, best_f1, best_ep


def grid_for(kind):
    g = dict(GRID)
    if kind in ("img", "txt", "concat_linear"):
        g["dropout"] = [0.0]
    return [dict(zip(g, v)) for v in itertools.product(*g.values())]


def tune(data, dev):
    cfgs = {}
    for kind in HEADS:
        scored = []
        for cfg in grid_for(kind):
            _, f1, _ = train_head(kind, data, cfg, 0, dev)
            scored.append((f1, cfg))
            print(f"Tune {kind:13s} {cfg} holdout_f1={f1:.4f}", flush=True)
        cfgs[kind] = max(scored, key=lambda x: x[0])[1]

    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump({**cfgs, "_holdout_id": data["holdout_id"]}, f, indent=1)
    return cfgs


def load_cfgs(data):
    if not os.path.exists(CONFIG_PATH):
        return None
    cfgs = json.load(open(CONFIG_PATH, encoding="utf-8"))
    return cfgs if cfgs.get("_holdout_id") == data["holdout_id"] else None


def run(data, dev, cfgs=None):
    cfgs = cfgs or load_cfgs(data) or tune(data, dev)
    os.makedirs(os.path.join(RESULTS_DIR, "heads"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "preds"), exist_ok=True)

    for kind in HEADS:
        for seed in SEEDS:
            model, hf1, ep = train_head(kind, data, cfgs[kind], seed, dev)
            res = {
                "model": kind,
                "seed": seed,
                "cfg": cfgs[kind],
                "best_epoch": ep,
                "holdout_id": data["holdout_id"],
                "holdout": compute_metrics(data["holdout"]["y"], predict_proba(model, data["holdout"], dev))
            }
            preds = {}
            for s in EVAL_SPLITS:
                prob = predict_proba(model, data[s], dev)
                res[s] = compute_metrics(data[s]["y"], prob)
                preds[s] = prob

            with open(f"{RESULTS_DIR}/heads/{kind}_seed{seed}.json", "w", encoding="utf-8") as f:
                json.dump(res, f, indent=1)
            np.savez(f"{RESULTS_DIR}/preds/heads_{kind}_seed{seed}.npz", **preds)
            print(f"{kind:13s} seed={seed} " + "  ".join(
                f"{s}_f1={res[s]['macro_f1']:.3f} acc={res[s]['acc']:.3f}" for s in ["holdout"] + EVAL_SPLITS),
                flush=True)


def summary(results_dir=RESULTS_DIR):
    out = {}
    for kind in HEADS:
        runs = [json.load(open(f"{results_dir}/heads/{kind}_seed{s}.json")) for s in SEEDS]
        out[kind] = {}
        for s in ["holdout"] + EVAL_SPLITS:
            for m in ("macro_f1", "weighted_f1", "acc"):
                v = np.array([r[s][m] for r in runs])
                out[kind][f"{s}_{m}"] = {
                    "mean": float(v.mean()),
                    "std": float(v.std(ddof=1)),
                    "n": len(v)
                }

    summary_file = os.path.join(results_dir, "heads_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    print("\n--- Phase 2 Heads Summary (Mean ± Std over 5 seeds) ---")
    for kind, r in out.items():
        val_str = f"val F1: {r['validation_macro_f1']['mean']:.3f}±{r['validation_macro_f1']['std']:.3f}  acc: {r['validation_acc']['mean']:.3f}±{r['validation_acc']['std']:.3f}"
        test_str = f"test F1: {r['test_macro_f1']['mean']:.3f}±{r['test_macro_f1']['std']:.3f}  acc: {r['test_acc']['mean']:.3f}±{r['test_acc']['std']:.3f}"
        print(f"{kind:14s}  {val_str}  |  {test_str}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.tune or a.run:
        features_data = load_features()
        tuned_cfgs = tune(features_data, dev) if a.tune else load_cfgs(features_data)
        if a.run and tuned_cfgs is None:
            print("configs/heads.json missing or stale: re-tuning...")
            tuned_cfgs = tune(features_data, dev)
        if a.run:
            run(features_data, dev, tuned_cfgs)
    if a.summary or a.run:
        summary()
