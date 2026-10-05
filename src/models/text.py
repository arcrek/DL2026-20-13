import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
import transformers

transformers.logging.set_verbosity_error()

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data import DATA_DIR, FEATURE_DIR, load_jsonl
from results import RESULTS_DIR, save_results

MODEL_NAME = "bert-base-uncased"
NUM_CLASSES = 3


class TextMemeDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        return {
            "id": str(r["id"]),
            "text": str(r.get("text", "")),
            "label": int(r["label"]),
            "sarcasm": str(r.get("sarcasm", "not_sarcastic"))
        }


class BertMemeClassifier(nn.Module):
    def __init__(self, model_name=MODEL_NAME, num_classes=NUM_CLASSES, dropout=0.2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def backbone_params(self):
        return [p for n, p in self.named_parameters() if n.startswith("bert.") and p.requires_grad]

    def head_params(self):
        return [p for n, p in self.named_parameters() if not n.startswith("bert.") and p.requires_grad]

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_rep = outputs.pooler_output
        return self.classifier(self.drop(cls_rep))


def make_collate_fn(tokenizer, max_length=128):
    def collate_fn(batch):
        ids = [item["id"] for item in batch]
        texts = [item["text"] for item in batch]
        labels = [item["label"] for item in batch]
        sarcasms = [item["sarcasm"] for item in batch]

        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        return {
            "ids": ids,
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long),
            "sarcasms": sarcasms
        }
    return collate_fn


@torch.no_grad()
def evaluate_split(model, loader, device):
    model.eval()
    all_probs, all_ys, all_ids, all_sarcasms = [], [], [], []
    for batch in loader:
        ids = batch["ids"]
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        autocast_dtype = torch.bfloat16 if (torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()) else torch.float16
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=(device == "cuda")):
            logits = model(input_ids, mask)
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()

        all_probs.append(probs)
        all_ys.append(labels.cpu().numpy())
        all_ids.extend(ids)
        all_sarcasms.extend(batch["sarcasms"])

    probs_arr = np.concatenate(all_probs, axis=0)
    ys_arr = np.concatenate(all_ys, axis=0)
    preds = np.argmax(probs_arr, axis=-1)

    macro_f1 = float(f1_score(ys_arr, preds, average="macro", zero_division=0))
    acc = float(accuracy_score(ys_arr, preds))
    return {
        "macro_f1": macro_f1,
        "acc": acc,
        "probs": probs_arr,
        "y_true": ys_arr,
        "ids": all_ids,
        "sarcasm": all_sarcasms
    }


def train_seed(seed, cfg, device="cuda"):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    ho_path = os.path.join(cfg.get("feature_dir", FEATURE_DIR), "train_holdout.json")
    if not os.path.exists(ho_path):
        raise FileNotFoundError(f"Missing holdout file: {ho_path}")

    with open(ho_path, "r", encoding="utf-8") as f:
        ho = json.load(f)

    train_rows = {r["id"]: r for r in load_jsonl("train", data_dir=cfg.get("data_dir", DATA_DIR))}
    fit_rows = [train_rows[i] for i in ho["fit"]]
    holdout_rows = [train_rows[i] for i in ho["holdout"]]
    test_rows = load_jsonl("test", data_dir=cfg.get("data_dir", DATA_DIR))

    class_weights = torch.tensor(ho["class_weights"], dtype=torch.float).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    collate = make_collate_fn(tokenizer, max_length=cfg.get("max_length", 128))

    g = torch.Generator().manual_seed(seed)
    fit_loader = DataLoader(
        TextMemeDataset(fit_rows),
        batch_size=cfg["batch"],
        shuffle=True,
        collate_fn=collate,
        generator=g,
        num_workers=cfg.get("workers", 2)
    )
    holdout_loader = DataLoader(
        TextMemeDataset(holdout_rows),
        batch_size=64,
        shuffle=False,
        collate_fn=collate,
        num_workers=cfg.get("workers", 2)
    )
    test_loader = DataLoader(
        TextMemeDataset(test_rows),
        batch_size=64,
        shuffle=False,
        collate_fn=collate,
        num_workers=cfg.get("workers", 2)
    )

    model = BertMemeClassifier(MODEL_NAME, num_classes=NUM_CLASSES, dropout=cfg.get("dropout", 0.2)).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.backbone_params(), "lr": cfg["lr_backbone"]},
        {"params": model.head_params(), "lr": cfg["lr_head"]}
    ], weight_decay=cfg.get("weight_decay", 0.01))

    total_steps = cfg["epochs"] * len(fit_loader)
    warmup_steps = int(0.10 * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: min(1.0, (s + 1) / max(1, warmup_steps)) * max(0.0, (total_steps - s) / max(1, total_steps))
    )

    best_f1 = -1.0
    best_epoch = -1
    best_state = None

    total_batches = len(fit_loader)
    for ep in range(cfg["epochs"]):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(fit_loader, 1):
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            autocast_dtype = torch.bfloat16 if (torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()) else torch.float16
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=(device == "cuda")):
                logits = model(input_ids, mask)
                loss = loss_fn(logits.float(), labels)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            log_interval = 15 if device == "cpu" else 30
            if step % log_interval == 0 or step == total_batches:
                pct = (step / total_batches) * 100
                print(f"[Seed {seed} | Ep {ep+1:02d}/{cfg['epochs']:02d}] Batch {step:03d}/{total_batches} ({pct:5.1f}%) | Loss: {total_loss / step:.4f}", flush=True)

        avg_loss = total_loss / total_batches
        eval_ho = evaluate_split(model, holdout_loader, device)
        ho_f1 = eval_ho["macro_f1"]
        ho_acc = eval_ho["acc"]

        print(f"[Seed {seed}] Epoch {ep + 1:02d}/{cfg['epochs']:02d} | Train Loss: {avg_loss:.4f} | Holdout F1: {ho_f1:.4f} | Holdout Acc: {ho_acc:.4f}", flush=True)

        if ho_f1 > best_f1:
            best_f1 = ho_f1
            best_epoch = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(f"[Seed {seed}] Best Epoch: {best_epoch} | Holdout Macro-F1: {best_f1:.4f}")

    model.load_state_dict(best_state)
    test_eval = evaluate_split(model, test_loader, device)

    res_dir = cfg.get("results_dir", RESULTS_DIR)
    out_path = save_results(
        config="text",
        seed=seed,
        ids=test_eval["ids"],
        y_true=test_eval["y_true"],
        probs=test_eval["probs"],
        sarcasm=test_eval["sarcasm"],
        split="test",
        extra={
            "model": MODEL_NAME,
            "best_epoch": best_epoch,
            "holdout_macro_f1": best_f1,
            "lr_backbone": cfg["lr_backbone"],
            "lr_head": cfg["lr_head"],
        },
        results_dir=res_dir
    )

    print(f"[Seed {seed}] Saved result: {out_path} | Test Macro-F1: {test_eval['macro_f1']:.4f} | Test Acc: {test_eval['acc']:.4f}")

    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return test_eval["macro_f1"], test_eval["acc"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr-backbone", type=float, default=1.5e-5)
    parser.add_argument("--lr-head", type=float, default=5e-4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    cfg = {
        "epochs": args.epochs,
        "batch": args.batch,
        "lr_backbone": args.lr_backbone,
        "lr_head": args.lr_head,
        "max_length": args.max_length,
        "workers": args.workers,
        "dropout": 0.2,
        "weight_decay": 0.01
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")
    print(f"Config: {cfg}")
    print(f"Seeds: {args.seeds}")

    f1_list, acc_list = [], []
    for s in args.seeds:
        f1, acc = train_seed(s, cfg, device=device)
        f1_list.append(f1)
        acc_list.append(acc)

    print(f"Macro-F1 per seed: {[round(x, 4) for x in f1_list]}")
    print(f"Accuracy per seed: {[round(x, 4) for x in acc_list]}")
    print(f"Macro-F1 mean: {np.mean(f1_list):.4f} +/- {np.std(f1_list):.4f}")
    print(f"Accuracy mean: {np.mean(acc_list):.4f} +/- {np.std(acc_list):.4f}")


if __name__ == "__main__":
    main()
