"""Cross-Attention Multimodal Fusion module (Branch 3B: both_cross_attn).

Re-exports CrossAttentionFusionModel and multimodal components from src.models.fusion,
and provides the dedicated training and evaluation pipeline for Cross-Attention.
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data import DATA_DIR, FEATURE_DIR, load_jsonl
from models.fusion import (
    NUM_CLASSES,
    TEXT_MODEL_NAME,
    CrossAttentionFusionModel,
    FusionCollate,
    MultimodalMemeDataset,
    ResNet50Backbone,
    make_fusion_collate_fn,
    safe_open_image,
)
from results import RESULTS_DIR, save_results

try:
    from preprocessing import is_valid_image
except ImportError:
    from PIL import Image

    def is_valid_image(path):
        if not path or not os.path.exists(str(path)):
            return False
        try:
            with Image.open(str(path)) as img:
                img.verify()
            with Image.open(str(path)) as img:
                img.convert("RGB")
            return True
        except Exception:
            return False

CONFIG_NAME = "both_cross_attn"


def make_multimodal_collate_fn(tokenizer, image_transform=None, max_length=128):
    if image_transform is None:
        from torchvision import transforms
        image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    return make_fusion_collate_fn(tokenizer, image_transform=image_transform, max_length=max_length)


def build_loss_and_optimizer(model, cfg, class_weights, total_steps, device):
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW([
        {"params": model.backbone_params(), "lr": cfg["lr_backbone"]},
        {"params": model.head_params(), "lr": cfg["lr_head"]}
    ], weight_decay=cfg.get("weight_decay", 0.01))

    warmup_steps = int(0.10 * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: min(1.0, (s + 1) / max(1, warmup_steps)) * max(0.0, (total_steps - s) / max(1, total_steps))
    )
    return loss_fn, optimizer, scheduler


@torch.no_grad()
def evaluate(model, loader, device, loss_fn=None):
    model.eval()
    all_probs, all_ys, all_ids, all_sarcasms = [], [], [], []
    total_loss = 0.0
    autocast_dtype = (
        torch.bfloat16
        if (torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())
        else torch.float16
    )

    for batch in loader:
        ids = batch["ids"]
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.autocast("cuda", dtype=autocast_dtype, enabled=(str(device).startswith("cuda"))):
            logits = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=mask)
            if loss_fn is not None:
                loss = loss_fn(logits.float(), labels)
                total_loss += loss.item()

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
    avg_loss = total_loss / len(loader) if loss_fn is not None else 0.0

    return {
        "macro_f1": macro_f1,
        "acc": acc,
        "loss": avg_loss,
        "probs": probs_arr,
        "y_true": ys_arr,
        "ids": all_ids,
        "sarcasm": all_sarcasms,
    }


evaluate_split = evaluate


def train_epoch(model, loader, loss_fn, optimizer, scheduler, device, epoch_idx, total_epochs, seed):
    model.train()
    total_loss = 0.0
    total_batches = len(loader)
    autocast_dtype = (
        torch.bfloat16
        if (torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())
        else torch.float16
    )

    for step, batch in enumerate(loader, 1):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=(str(device).startswith("cuda"))):
            logits = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=mask)
            loss = loss_fn(logits.float(), labels)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        log_interval = 15 if str(device) == "cpu" else 30
        if step % log_interval == 0 or step == total_batches:
            pct = (step / total_batches) * 100
            print(
                f"[Cross-Attn | Seed {seed} | Ep {epoch_idx+1:02d}/{total_epochs:02d}] "
                f"Batch {step:03d}/{total_batches} ({pct:5.1f}%) | Loss: {total_loss / step:.4f}",
                flush=True,
            )

    return total_loss / total_batches


def train_loop(seed, cfg, device="cuda", data_df=None):
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

    data_dir = cfg.get("data_dir", DATA_DIR)

    if data_df is not None:
        records = data_df.to_dict("records")
        train_rows = {str(r.get("id", i)): r for i, r in enumerate(records)}
    else:
        train_rows = {r["id"]: r for r in load_jsonl("train", data_dir=data_dir)}

    fit_rows = [train_rows[i] for i in ho["fit"] if i in train_rows]
    holdout_rows = [train_rows[i] for i in ho["holdout"] if i in train_rows]
    test_rows = load_jsonl("test", data_dir=data_dir)

    # Preprocessing integrity check: filter corrupted/missing images
    if is_valid_image is not None:
        bad_fit = [
            r["id"]
            for r in fit_rows
            if not is_valid_image(r.get("image_path", os.path.join(data_dir, r.get("img", ""))))
        ]
        if bad_fit:
            print(f"[Cross-Attn | Seed {seed}] Preprocessing: filtered {len(bad_fit)} corrupted/missing images from fit set.")
            fit_rows = [r for r in fit_rows if r["id"] not in bad_fit]

    class_weights = torch.tensor(ho["class_weights"], dtype=torch.float)

    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    image_transform = cfg.get("image_transform", None)
    collate = make_fusion_collate_fn(tokenizer, image_transform=image_transform, max_length=cfg.get("max_length", 128))

    g = torch.Generator().manual_seed(seed)
    fit_loader = DataLoader(
        MultimodalMemeDataset(fit_rows, data_dir=data_dir),
        batch_size=cfg["batch"],
        shuffle=True,
        collate_fn=collate,
        generator=g,
        num_workers=cfg.get("workers", 2),
    )
    holdout_loader = DataLoader(
        MultimodalMemeDataset(holdout_rows, data_dir=data_dir),
        batch_size=cfg.get("eval_batch", 32),
        shuffle=False,
        collate_fn=collate,
        num_workers=cfg.get("workers", 2),
    )
    test_loader = DataLoader(
        MultimodalMemeDataset(test_rows, data_dir=data_dir),
        batch_size=cfg.get("eval_batch", 32),
        shuffle=False,
        collate_fn=collate,
        num_workers=cfg.get("workers", 2),
    )

    model = CrossAttentionFusionModel(
        num_classes=NUM_CLASSES,
        text_model_name=TEXT_MODEL_NAME,
        attn_dim=cfg.get("attn_dim", 256),
        num_heads=cfg.get("num_heads", 4),
        dropout=cfg.get("dropout", 0.2),
        freeze_image=cfg.get("freeze_image", True),
    ).to(device)

    total_steps = cfg["epochs"] * len(fit_loader)
    loss_fn, optimizer, scheduler = build_loss_and_optimizer(model, cfg, class_weights, total_steps, device)

    best_f1 = -1.0
    best_epoch = -1
    best_state = None

    print(f"\n=== Training [both_cross_attn] Seed {seed} on {str(device).upper()} ===")
    for ep in range(cfg["epochs"]):
        avg_loss = train_epoch(model, fit_loader, loss_fn, optimizer, scheduler, device, ep, cfg["epochs"], seed)
        eval_ho = evaluate(model, holdout_loader, device, loss_fn)
        ho_f1 = eval_ho["macro_f1"]
        ho_acc = eval_ho["acc"]

        print(
            f"[Cross-Attn | Seed {seed}] Epoch {ep + 1:02d}/{cfg['epochs']:02d} | "
            f"Train Loss: {avg_loss:.4f} | Holdout F1: {ho_f1:.4f} | Holdout Acc: {ho_acc:.4f}",
            flush=True,
        )

        if ho_f1 > best_f1:
            best_f1 = ho_f1
            best_epoch = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(f"[Cross-Attn | Seed {seed}] Best Epoch: {best_epoch} | Best Holdout Macro-F1: {best_f1:.4f}")

    model.load_state_dict(best_state)
    test_eval = evaluate(model, test_loader, device)

    res_dir = cfg.get("results_dir", RESULTS_DIR)
    out_path = save_results(
        config=CONFIG_NAME,
        seed=seed,
        ids=test_eval["ids"],
        y_true=test_eval["y_true"],
        probs=test_eval["probs"],
        sarcasm=test_eval["sarcasm"],
        split="test",
        extra={
            "model": "CrossAttentionFusionModel",
            "text_backbone": TEXT_MODEL_NAME,
            "image_backbone": "resnet50",
            "best_epoch": best_epoch,
            "holdout_macro_f1": best_f1,
            "lr_backbone": cfg["lr_backbone"],
            "lr_head": cfg["lr_head"],
            "attn_dim": cfg.get("attn_dim", 256),
        },
        results_dir=res_dir,
    )

    print(
        f"[Cross-Attn | Seed {seed}] Saved result: {out_path} | "
        f"Test Macro-F1: {test_eval['macro_f1']:.4f} | Test Acc: {test_eval['acc']:.4f}"
    )

    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return test_eval["macro_f1"], test_eval["acc"]


def train_seed(*args, **kwargs):
    """Train Cross-Attention model for a single seed.

    Supports signatures:
    - train_seed(seed, cfg, device="cuda", data_df=None)
    - train_seed(config_name, seed, cfg, device="cuda", data_df=None)
    - train_seed(seed=0, cfg=cfg, ...)
    """
    if "config_name" in kwargs:
        kwargs.pop("config_name")
    if "config" in kwargs:
        kwargs.pop("config")
    if args and isinstance(args[0], str):
        # First arg was config_name e.g. "both_cross_attn"
        args = args[1:]
    return train_loop(*args, **kwargs)


class CrossAttentionArgs:
    """Hardcoded arguments and hyperparameters for Cross-Attention Multimodal Fusion (Role D)."""

    def __init__(
        self,
        seeds=(0, 1, 2),
        epochs=5,
        batch_size=16,
        eval_batch_size=32,
        lr_backbone=1.5e-5,
        lr_head=5e-4,
        max_length=128,
        attn_dim=256,
        num_heads=4,
        dropout=0.2,
        workers=2,
        freeze_image=True,
        weight_decay=0.01,
        data_dir=DATA_DIR,
        holdout_file=None,
        results_dir=None,
    ):
        self.seeds = list(seeds)
        self.epochs = epochs
        self.batch_size = batch_size
        self.batch = batch_size
        self.eval_batch_size = eval_batch_size
        self.eval_batch = eval_batch_size
        self.lr_backbone = lr_backbone
        self.lr_head = lr_head
        self.max_length = max_length
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.workers = workers
        self.freeze_image = freeze_image
        self.weight_decay = weight_decay
        self.data_dir = data_dir
        self.holdout_file = holdout_file or os.path.join(FEATURE_DIR, "train_holdout.json")
        self.results_dir = results_dir or os.environ.get("RESULTS_DIR", RESULTS_DIR)


def run_training(args=None, device=None, data_df=None):
    """Execute Cross-Attention training pipeline with hardcoded or user-supplied arguments."""
    if args is None:
        args = CrossAttentionArgs()
    elif isinstance(args, dict):
        args = CrossAttentionArgs(**args)

    cfg = {
        "epochs": getattr(args, "epochs", 5),
        "batch": getattr(args, "batch_size", getattr(args, "batch", 16)),
        "eval_batch": getattr(args, "eval_batch_size", getattr(args, "eval_batch", 32)),
        "lr_backbone": getattr(args, "lr_backbone", 1.5e-5),
        "lr_head": getattr(args, "lr_head", 5e-4),
        "max_length": getattr(args, "max_length", 128),
        "attn_dim": getattr(args, "attn_dim", 256),
        "num_heads": getattr(args, "num_heads", 4),
        "dropout": getattr(args, "dropout", 0.2),
        "workers": getattr(args, "workers", 2),
        "freeze_image": getattr(args, "freeze_image", True),
        "weight_decay": getattr(args, "weight_decay", 0.01),
        "data_dir": getattr(args, "data_dir", DATA_DIR),
        "feature_dir": os.path.dirname(getattr(args, "holdout_file", os.path.join(FEATURE_DIR, "train_holdout.json"))),
        "holdout_file": getattr(args, "holdout_file", os.path.join(FEATURE_DIR, "train_holdout.json")),
        "results_dir": getattr(args, "results_dir", RESULTS_DIR),
    }

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    seeds = list(getattr(args, "seeds", [0, 1, 2]))
    print(f"Device: {str(device).upper()}")
    print("Cross-Attention Fusion Configuration (hardcoded arguments):")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print(f"Seeds: {seeds}")

    # Data verification with preprocessing filter
    data_dir = cfg["data_dir"]
    if data_df is not None:
        print(f"Preprocessing integration: using preprocessed DataFrame ({len(data_df)} records).")
    else:
        raw_train = load_jsonl("train", data_dir=str(data_dir))
        corrupt = [
            r["id"]
            for r in raw_train
            if not is_valid_image(r.get("image_path", os.path.join(data_dir, r.get("img", ""))))
        ]
        if corrupt:
            print(f"Notice: {len(corrupt)} corrupted/missing images identified via preprocessing filter.")
        else:
            print(f"Data verification: All {len(raw_train)} training images validated successfully.")

    f1_list, acc_list = [], []
    for s in seeds:
        f1, acc = train_loop(s, cfg, device=device, data_df=data_df)
        f1_list.append(f1)
        acc_list.append(acc)

    print("\n" + "=" * 60)
    print("CROSS-ATTENTION MULTIMODAL FUSION TRAINING COMPLETED")
    print("=" * 60)
    print(f"Macro-F1 per seed: {[round(x, 4) for x in f1_list]}")
    print(f"Accuracy per seed: {[round(x, 4) for x in acc_list]}")
    print(f"Macro-F1 mean +/- std: {np.mean(f1_list):.4f} +/- {np.std(f1_list):.4f}")
    print(f"Accuracy mean +/- std: {np.mean(acc_list):.4f} +/- {np.std(acc_list):.4f}")

    return {
        "f1_list": f1_list,
        "acc_list": acc_list,
        "f1_mean": float(np.mean(f1_list)),
        "f1_std": float(np.std(f1_list)),
        "acc_mean": float(np.mean(acc_list)),
        "acc_std": float(np.std(acc_list)),
    }


def main(args=None):
    """Entry point supporting both CLI parsing and hardcoded argument execution."""
    if args is not None:
        return run_training(args)

    if len(sys.argv) > 1 and any(arg.startswith("--") for arg in sys.argv[1:]):
        parser = argparse.ArgumentParser(description="Train Cross-Attention Multimodal Fusion model.")
        parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
        parser.add_argument("--epochs", type=int, default=5)
        parser.add_argument("--batch", type=int, default=16)
        parser.add_argument("--eval-batch", type=int, default=32)
        parser.add_argument("--lr-backbone", type=float, default=1.5e-5)
        parser.add_argument("--lr-head", type=float, default=5e-4)
        parser.add_argument("--max-length", type=int, default=128)
        parser.add_argument("--attn-dim", type=int, default=256)
        parser.add_argument("--num-heads", type=int, default=4)
        parser.add_argument("--dropout", type=float, default=0.2)
        parser.add_argument("--workers", type=int, default=2)
        parser.add_argument("--freeze-image", action="store_true", default=True)
        parsed_args = parser.parse_args()
        return run_training(parsed_args)
    else:
        return run_training(CrossAttentionArgs())


if __name__ == "__main__":
    main()
