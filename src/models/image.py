import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from data import DATA_DIR, FEATURE_DIR, load_jsonl


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
        with Image.open(self.data_dir / row["img"]) as image:
            image = image.convert("RGB")
        return self.transform(image), int(row["label"])


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
    ys, preds = [], []

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            ys.extend(labels.tolist())
            preds.extend(logits.argmax(1).cpu().tolist())

    return {
        "accuracy": float(accuracy_score(ys, preds)),
        "macro_f1": float(f1_score(ys, preds, average="macro", zero_division=0)),
    }


def train_epoch(model, loader, optimizer, loss_fn, device, frozen):
    model.train()
    if frozen:
        model.net.eval()
        model.net.fc.train()

    total_loss = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        total += len(labels)

    return total_loss / total


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
    train_loader = DataLoader(
        MemeDataset(fit, data_dir, train_tf),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        generator=generator,
    )
    val_loader = DataLoader(
        MemeDataset(val, data_dir, eval_tf),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    model = ImageModel().to(device)
    best_f1, best_state, best_info = -1, None, {}
    history = []

    # Stage 1: train the new classifier; Stage 2: fine-tune ResNet at lower LR.
    stages = [
        ("head_only", args.head_epochs, True, 0.0, 1e-3),
        ("fine_tune", args.finetune_epochs, False, 1e-5, 1e-4),
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
                f"seed={seed} {name} {epoch + 1}/{epochs} "
                f"loss={loss:.4f} holdout_f1={metrics['macro_f1']:.4f}"
            )

    out = ROOT / "checkpoints"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"seed": seed, "state_dict": best_state, "best": best_info},
        out / f"image_seed{seed}.pt",
    )

    out = ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"image_train_seed{seed}.json").open("w") as file:
        json.dump({"seed": seed, "best": best_info, "history": history}, file, indent=2)

    return best_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument(
        "--holdout-file",
        default=str(Path(FEATURE_DIR) / "train_holdout.json"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--head-epochs", type=int, default=3)
    parser.add_argument("--finetune-epochs", type=int, default=5)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    rows = load_jsonl("train", data_dir=str(data_dir))

    bad = [
        (row["id"], row["img"])
        for row in rows
        if not valid_image(data_dir / row["img"])
    ]
    if bad:
        raise RuntimeError(
            f"Ảnh train bị thiếu/hỏng: {bad[:20]}. "
            "Hãy làm sạch dữ liệu chung rồi tạo lại holdout và class weights."
        )

    with open(args.holdout_file, encoding="utf-8") as file:
        holdout = json.load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scores = [
        train_seed(seed, rows, holdout, data_dir, device, args)
        for seed in args.seeds
    ]
    print("Holdout Macro-F1:", scores)
    print(f"Mean ± std: {np.mean(scores):.4f} ± {np.std(scores):.4f}")


if __name__ == "__main__":
    main()