import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFile, ImageOps
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data import DATA_DIR, FEATURE_DIR, load_jsonl
from results import RESULTS_DIR, save_results


class ResizeAndPad:
    def __call__(self, image):
        image = ImageOps.contain(
            image.convert("RGB"),
            (224, 224),
            method=Image.Resampling.BILINEAR,
        )
        canvas = Image.new("RGB", (224, 224), (124, 116, 104))
        canvas.paste(
            image,
            ((224 - image.width) // 2, (224 - image.height) // 2),
        )
        return canvas


def valid_image(path):
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.convert("RGB").load()
        return True
    except Exception:
        return False


class MemeDataset(Dataset):
    def __init__(self, rows, data_dir, transform):
        self.rows = rows
        self.data_dir = Path(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        img_name = row.get("img") or row.get("image_path") or row.get("image") or ""
        img_path = self.data_dir / img_name if not os.path.isabs(str(img_name)) else Path(img_name)
        try:
            with Image.open(img_path) as image:
                img_rgb = image.convert("RGB")
        except Exception:
            img_rgb = Image.new("RGB", (224, 224), (124, 116, 104))
        return (
            self.transform(img_rgb),
            int(row["label"]),
            str(row["id"]),
            str(row.get("sarcasm", "not_sarcastic")),
        )


class ImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
        )
        n = self.net.fc.in_features
        self.net.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(n, 3))

    def forward(self, images):
        return self.net(images)


def evaluate(model, loader, device):
    model.eval()
    ys, preds, all_probs, all_ids, all_sarcasms = [], [], [], [], []

    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(device)
            labels = batch[1]
            logits = model(images)
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            all_probs.append(probs)
            ys.extend(labels.tolist())
            preds.extend(logits.argmax(1).cpu().tolist())
            if len(batch) > 2:
                all_ids.extend(batch[2])
            if len(batch) > 3:
                all_sarcasms.extend(batch[3])

    probs_arr = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, 3))
    ys_arr = np.array(ys, dtype=int)
    preds_arr = np.array(preds, dtype=int)

    return {
        "accuracy": float(accuracy_score(ys_arr, preds_arr)) if len(ys_arr) else 0.0,
        "macro_f1": float(f1_score(ys_arr, preds_arr, average="macro", zero_division=0)) if len(ys_arr) else 0.0,
        "probs": probs_arr,
        "y_true": ys_arr,
        "y_pred": preds_arr,
        "ids": all_ids,
        "sarcasm": all_sarcasms,
    }


def train_epoch(model, loader, optimizer, loss_fn, device, frozen):
    model.train()
    if frozen:
        model.net.eval()
        model.net.fc.train()

    total_loss = 0.0
    total = 0

    for batch in loader:
        images, labels = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        total += len(labels)

    return total_loss / total if total > 0 else 0.0


def train_seed(seed, rows, holdout, data_dir, device, args):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    by_id = {str(row["id"]): row for row in rows}
    fit = [by_id[str(i)] for i in holdout["fit"]]
    val = [by_id[str(i)] for i in holdout["holdout"]]

    weights = torch.tensor(holdout["class_weights"], dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))

    resize = ResizeAndPad()
    norm = transforms.Normalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    )
    train_tf = transforms.Compose([
        resize,
        transforms.ColorJitter(0.08, 0.08, 0.05),
        transforms.ToTensor(),
        norm,
    ])
    eval_tf = transforms.Compose([resize, transforms.ToTensor(), norm])

    generator = torch.Generator().manual_seed(seed)
    workers = getattr(args, "workers", 2)
    batch_size = getattr(args, "batch_size", 32)
    head_epochs = getattr(args, "head_epochs", 3)
    finetune_epochs = getattr(args, "finetune_epochs", 5)

    train_loader = DataLoader(
        MemeDataset(fit, data_dir, train_tf),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        generator=generator,
    )
    val_loader = DataLoader(
        MemeDataset(val, data_dir, eval_tf),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )

    model = ImageModel().to(device)
    best_f1, best_state, best_info = -1.0, None, {}
    history = []

    # Stage 1: train the new classifier; Stage 2: fine-tune ResNet at lower LR.
    stages = [
        ("head_only", head_epochs, True, 0.0, 1e-3),
        ("fine_tune", finetune_epochs, False, 1e-5, 1e-4),
    ]

    epoch_no = 0
    for name, epochs, frozen, backbone_lr, head_lr in stages:
        for pname, p in model.net.named_parameters():
            p.requires_grad = pname.startswith("fc.") or not frozen

        if frozen:
            optimizer = torch.optim.AdamW(
                model.net.fc.parameters(), lr=head_lr
            )
        else:
            backbone = [
                p for n, p in model.net.named_parameters()
                if not n.startswith("fc.")
            ]
            optimizer = torch.optim.AdamW([
                {"params": backbone, "lr": backbone_lr},
                {"params": model.net.fc.parameters(), "lr": head_lr},
            ])

        for epoch in range(epochs):
            epoch_no += 1
            loss = train_epoch(
                model, train_loader, optimizer, loss_fn, device, frozen
            )
            metrics = evaluate(model, val_loader, device)
            history.append({
                "epoch": epoch_no,
                "stage": name,
                "train_loss": loss,
                **metrics,
            })

            if metrics["macro_f1"] > best_f1:
                best_f1 = metrics["macro_f1"]
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                best_info = {
                    "epoch": epoch_no,
                    "stage": name,
                    **metrics,
                }

            print(
                f"[Seed {seed}] {name} {epoch + 1}/{epochs} | "
                f"Loss: {loss:.4f} | Holdout Macro-F1: {metrics['macro_f1']:.4f}"
            )

    # Save checkpoint
    out_ckp = ROOT / "checkpoints"
    out_ckp.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"seed": seed, "state_dict": best_state, "best": best_info},
        out_ckp / f"image_seed{seed}.pt",
    )

    # Evaluate on test split and save canonical results contract
    model.load_state_dict(best_state)
    test_rows = load_jsonl("test", data_dir=str(data_dir))
    test_loader = DataLoader(
        MemeDataset(test_rows, data_dir, eval_tf),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )
    test_eval = evaluate(model, test_loader, device)

    res_dir = getattr(args, "results_dir", None) or os.environ.get("RESULTS_DIR", str(ROOT / "results"))
    out_path = save_results(
        config="image",
        seed=seed,
        ids=test_eval["ids"],
        y_true=test_eval["y_true"],
        probs=test_eval["probs"],
        sarcasm=test_eval["sarcasm"],
        split="test",
        extra={
            "model": "resnet50",
            "best_epoch": best_info.get("epoch", 0),
            "holdout_macro_f1": best_f1,
            "head_epochs": head_epochs,
            "finetune_epochs": finetune_epochs,
        },
        results_dir=res_dir,
    )

    print(
        f"[Seed {seed}] Saved result: {out_path} | "
        f"Test Macro-F1: {test_eval['macro_f1']:.4f} | Test Acc: {test_eval['accuracy']:.4f}"
    )

    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return test_eval["macro_f1"], test_eval["accuracy"]


class ImageArgs:
    """Hardcoded arguments and hyperparameters for the ResNet50 baseline."""

    def __init__(
        self,
        seeds=(0, 1, 2),
        data_dir=DATA_DIR,
        holdout_file=None,
        batch_size=32,
        workers=2,
        head_epochs=3,
        finetune_epochs=5,
        results_dir=None,
    ):
        self.seeds = list(seeds)
        self.data_dir = data_dir
        self.holdout_file = holdout_file or str(Path(FEATURE_DIR) / "train_holdout.json")
        self.batch_size = batch_size
        self.workers = workers
        self.head_epochs = head_epochs
        self.finetune_epochs = finetune_epochs
        self.results_dir = results_dir or os.environ.get("RESULTS_DIR", str(ROOT / "results"))


def run_training(args=None, device=None):
    """Execute training pipeline with hardcoded or user-supplied arguments."""
    if args is None:
        args = ImageArgs()
    elif isinstance(args, dict):
        args = ImageArgs(**args)

    data_dir = Path(args.data_dir)
    rows = load_jsonl("train", data_dir=str(data_dir))

    bad = [
        (row["id"], row["img"])
        for row in rows
        if not valid_image(data_dir / row["img"])
    ]
    if bad:
        print(f"Notice: {len(bad)} corrupted/missing images identified in training rows.")

    with open(args.holdout_file, encoding="utf-8") as file:
        holdout = json.load(file)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print("Image Baseline Configuration (hardcoded arguments):")
    print(f"  seeds: {args.seeds}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  head_epochs: {args.head_epochs}")
    print(f"  finetune_epochs: {args.finetune_epochs}")
    print(f"  workers: {args.workers}")
    print(f"  data_dir: {args.data_dir}")

    f1_list, acc_list = [], []
    for seed in args.seeds:
        f1, acc = train_seed(seed, rows, holdout, data_dir, device, args)
        f1_list.append(f1)
        acc_list.append(acc)

    print("\n" + "=" * 50)
    print("IMAGE BASELINE TRAINING COMPLETED")
    print("=" * 50)
    print(f"Macro-F1 per seed: {[round(x, 4) for x in f1_list]}")
    print(f"Accuracy per seed: {[round(x, 4) for x in acc_list]}")
    print(f"Macro-F1 mean ± std: {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")
    print(f"Accuracy mean ± std: {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")

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

    # If CLI flags were provided, parse them; otherwise fall back to hardcoded defaults
    if len(sys.argv) > 1 and any(arg.startswith("--") for arg in sys.argv[1:]):
        parser = argparse.ArgumentParser(description="Image Baseline Trainer (ResNet50)")
        parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
        parser.add_argument("--data-dir", default=DATA_DIR)
        parser.add_argument(
            "--holdout-file",
            default=str(Path(FEATURE_DIR) / "train_holdout.json"),
        )
        parser.add_argument("--batch-size", type=int, default=32)
        parser.add_argument("--workers", type=int, default=2)
        parser.add_argument("--head-epochs", type=int, default=3)
        parser.add_argument("--finetune-epochs", type=int, default=5)
        parsed_args = parser.parse_args()
        return run_training(parsed_args)
    else:
        # Default hardcoded arguments
        return run_training(ImageArgs())


if __name__ == "__main__":
    main()