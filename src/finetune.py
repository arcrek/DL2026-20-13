"""Fine-tuned models for Memotion 7k sentiment analysis (GPU).

Entry point for 3 modes:
  python src/finetune.py --mode vit  --seeds 0 1 2   # ViT-B/16, image only
  python src/finetune.py --mode bert --seeds 0 1 2   # BERT-base, text only
  python src/finetune.py --mode clip --seeds 0 1 2   # CLIP ViT-L/14 + gated fusion, last N blocks unfrozen

Model selection uses train hold-out Macro F1. Validation and test are strictly for evaluation.
Per run: results/finetune/{mode}_seed{S}.json and results/preds/ft_{mode}_seed{S}.npz
"""
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor, ViTImageProcessor)

from data import DATA_DIR, load_jsonl

ImageFile.LOAD_TRUNCATED_IMAGES = True

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")
FEATURE_DIR = os.environ.get("FEATURE_DIR", "features")
EVAL_SPLITS = ["validation", "test"]
NUM_CLASSES = 3
NAMES = {
    "vit": "google/vit-base-patch16-224-in21k",
    "bert": "bert-base-uncased",
    "clip": "openai/clip-vit-large-patch14"
}
DEFAULTS = {
    "vit": dict(lr=3e-5, lr_head=1e-3, epochs=6, batch=32),
    "bert": dict(lr=3e-5, lr_head=1e-3, epochs=6, batch=32),
    "clip": dict(lr=1e-5, lr_head=1e-3, epochs=6, batch=32, unfreeze=2),
}


def safe_open_image(path):
    try:
        img = Image.open(path)
        img.load()
        return img.convert("RGB")
    except Exception:
        return Image.new("RGB", (224, 224), (128, 128, 128))


class MemeSet(Dataset):
    def __init__(self, rows, data_dir=DATA_DIR):
        self.rows = rows
        self.dir = data_dir

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, k):
        r = self.rows[k]
        img_path = os.path.join(self.dir, r["img"])
        img = safe_open_image(img_path)
        return img, r["text"], r["label"], r["id"]


class Net(nn.Module):
    """Wraps ViT (image), BERT (text), or CLIP (multimodal gated) with a 3-class classification head."""

    def __init__(self, mode, unfreeze=2, num_classes=NUM_CLASSES):
        super().__init__()
        self.mode = mode
        if mode == "vit":
            self.enc = AutoModel.from_pretrained(NAMES[mode])
            self.head = nn.Linear(self.enc.config.hidden_size, num_classes)
        elif mode == "bert":
            self.enc = AutoModel.from_pretrained(NAMES[mode])
            self.head = nn.Linear(self.enc.config.hidden_size, num_classes)
        else:
            self.enc = CLIPModel.from_pretrained(NAMES[mode])
            for p in self.enc.parameters():
                p.requires_grad = False
            for tower, proj in ((self.enc.vision_model, self.enc.visual_projection),
                                (self.enc.text_model, self.enc.text_projection)):
                blocks = tower.encoder.layers[-unfreeze:] if unfreeze else []
                post_ln = getattr(tower, "post_layernorm", None) or tower.final_layer_norm
                for m in list(blocks) + [proj, post_ln]:
                    for p in m.parameters():
                        p.requires_grad = True
            d = self.enc.config.projection_dim
            self.pi = nn.Linear(d, 256)
            self.pt = nn.Linear(d, 256)
            self.gate = nn.Linear(2 * d, 256)
            self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(256, num_classes))

    def head_params(self):
        return [p for n, p in self.named_parameters() if not n.startswith("enc.") and p.requires_grad]

    def backbone_params(self):
        return [p for n, p in self.named_parameters() if n.startswith("enc.") and p.requires_grad]

    def forward(self, px, ids, mask):
        if self.mode == "vit":
            return self.head(self.enc(pixel_values=px).pooler_output)
        if self.mode == "bert":
            return self.head(self.enc(input_ids=ids, attention_mask=mask).pooler_output)

        i = self.enc.visual_projection(self.enc.vision_model(pixel_values=px).pooler_output)
        t = self.enc.text_projection(self.enc.text_model(input_ids=ids, attention_mask=mask).pooler_output)
        i = nn.functional.normalize(i, dim=-1)
        t = nn.functional.normalize(t, dim=-1)
        g = torch.sigmoid(self.gate(torch.cat([i, t], -1)))
        fused = g * torch.tanh(self.pi(i)) + (1.0 - g) * torch.tanh(self.pt(t))
        return self.head(fused)


def make_collate(mode, proc, tok):
    def collate(batch):
        imgs, texts, ys, ids = zip(*batch)
        out = {
            "y": torch.tensor(ys, dtype=torch.long),
            "ids": list(ids)
        }
        if mode != "bert":
            out["px"] = proc(images=list(imgs), return_tensors="pt")["pixel_values"]
        else:
            out["px"] = None

        if mode != "vit":
            max_len = 77 if mode == "clip" else 128
            enc = tok(list(texts), return_tensors="pt", padding=True, truncation=True, max_length=max_len)
            out["input_ids"] = enc["input_ids"]
            out["mask"] = enc["attention_mask"]
        else:
            out["input_ids"] = None
            out["mask"] = None
        return out
    return collate


def to_dev(b, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}


@torch.no_grad()
def predict_proba(model, loader, dev):
    model.eval()
    all_probs, all_ys = [], []
    for b in loader:
        b = to_dev(b, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            logits = model(b["px"], b["input_ids"], b["mask"])
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        all_probs.append(probs)
        all_ys.append(b["y"].cpu().numpy())
    return np.concatenate(all_probs, axis=0), np.concatenate(all_ys, axis=0)


def compute_metrics(y_true, probas):
    preds = np.argmax(probas, axis=-1)
    return {
        "macro_f1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, preds, average="weighted", zero_division=0)),
        "acc": float(accuracy_score(y_true, preds))
    }


def run_seed(mode, seed, cfg, workers=4, dev="cuda"):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if mode == "clip":
        p = CLIPProcessor.from_pretrained(NAMES[mode])
        proc, tok = p.image_processor, p.tokenizer
    elif mode == "vit":
        proc, tok = ViTImageProcessor.from_pretrained(NAMES[mode]), None
    else:
        proc, tok = None, AutoTokenizer.from_pretrained(NAMES[mode])

    train_rows = {r["id"]: r for r in load_jsonl("train")}
    ho_path = os.path.join(FEATURE_DIR, "train_holdout.json")
    with open(ho_path, encoding="utf-8") as f:
        ho = json.load(f)

    sets = {
        "fit": [train_rows[i] for i in ho["fit"]],
        "holdout": [train_rows[i] for i in ho["holdout"]]
    }
    sets.update({s: load_jsonl(s) for s in EVAL_SPLITS})

    col = make_collate(mode, proc, tok)
    g = torch.Generator().manual_seed(seed)
    loader = lambda name, shuffle=False: DataLoader(
        MemeSet(sets[name]),
        batch_size=cfg["batch"] if shuffle else 64,
        shuffle=shuffle,
        collate_fn=col,
        num_workers=workers,
        generator=g if shuffle else None,
        persistent_workers=False
    )
    train_dl = loader("fit", shuffle=True)
    evals = {s: loader(s, shuffle=False) for s in ["holdout"] + EVAL_SPLITS}

    model = Net(mode, cfg.get("unfreeze", 2)).to(dev)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW([
        {"params": model.backbone_params(), "lr": cfg["lr"]},
        {"params": model.head_params(), "lr": cfg["lr_head"]}
    ], weight_decay=0.01)

    total_steps = cfg["epochs"] * len(train_dl)
    warmup_steps = int(0.1 * total_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / max(1, warmup_steps)) * max(0.0, (total_steps - s) / max(1, total_steps))
    )

    best_f1, best_ep, best_state = -1.0, -1, None
    for ep in range(cfg["epochs"]):
        model.train()
        for b in train_dl:
            b = to_dev(b, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                logits = model(b["px"], b["input_ids"], b["mask"])
            loss = loss_fn(logits.float(), b["y"])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

        val_probs, val_ys = predict_proba(model, evals["holdout"], dev)
        f1 = compute_metrics(val_ys, val_probs)["macro_f1"]
        print(f"FT {mode} seed={seed} ep={ep} holdout_f1={f1:.4f}", flush=True)
        if f1 > best_f1:
            best_f1, best_ep = f1, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    res = {
        "model": f"ft_{mode}",
        "seed": seed,
        "cfg": cfg,
        "best_epoch": best_ep
    }
    preds = {}
    for s in ["holdout"] + EVAL_SPLITS:
        probs, ys = predict_proba(model, evals[s], dev)
        res[s] = compute_metrics(ys, probs)
        preds[s] = probs

    os.makedirs(os.path.join(RESULTS_DIR, "finetune"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "preds"), exist_ok=True)
    with open(f"{RESULTS_DIR}/finetune/{mode}_seed{seed}.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    np.savez(f"{RESULTS_DIR}/preds/ft_{mode}_seed{seed}.npz", **preds)
    print(f"Done FT {mode} seed={seed} " + "  ".join(
        f"{s}_f1={res[s]['macro_f1']:.3f} acc={res[s]['acc']:.3f}" for s in ["holdout"] + EVAL_SPLITS),
        flush=True)

    del model, opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(NAMES.keys()), required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--unfreeze", type=int, help="CLIP mode: number of final transformer layers to unfreeze")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    cfg = dict(DEFAULTS[a.mode])
    for k in ("epochs", "batch", "unfreeze"):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for s in a.seeds:
        dest_json = f"{RESULTS_DIR}/finetune/{a.mode}_seed{s}.json"
        if os.path.exists(dest_json):
            print(f"Skipping {a.mode} seed {s} (already finished at {dest_json})")
            continue
        run_seed(a.mode, s, cfg, a.workers, dev)


if __name__ == "__main__":
    main()
