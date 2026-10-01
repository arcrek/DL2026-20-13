"""Extract and cache frozen CLIP ViT-L/14 image and text embeddings for Memotion 7k.

Outputs per split in features/:
  {split}_img.npy (float32, 768-d)
  {split}_txt.npy (float32, 768-d)
  {split}_ids.npy (str)
  {split}_labels.npy (int64, classes 0, 1, 2)

Run on GPU (Colab): python src/features.py [--splits train validation test] [--batch 128]
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image, ImageFile
from transformers import CLIPModel, CLIPProcessor

from data import DATA_DIR, SPLITS, load_jsonl

ImageFile.LOAD_TRUNCATED_IMAGES = True

MODEL = "openai/clip-vit-large-patch14"
OUT_DIR = os.environ.get("FEATURE_DIR", "features")
DIM = 768


def safe_open_image(path):
    try:
        img = Image.open(path)
        img.load()
        return img.convert("RGB")
    except Exception as e:
        # Graceful fallback for corrupted/truncated web images
        return Image.new("RGB", (224, 224), (128, 128, 128))


@torch.no_grad()
def extract_split(split, model, proc, device, batch=128, data_dir=DATA_DIR):
    rows = load_jsonl(split, data_dir)
    imgs, txts = [], []
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        images = [safe_open_image(os.path.join(data_dir, r["img"])) for r in chunk]
        px = proc(images=images, return_tensors="pt")["pixel_values"].to(device)
        tok = proc(text=[r["text"] for r in chunk], return_tensors="pt", padding=True,
                   truncation=True, max_length=77).to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
            # Manual projection for consistency across transformers versions
            iv = model.visual_projection(model.vision_model(pixel_values=px).pooler_output)
            tv = model.text_projection(model.text_model(
                input_ids=tok["input_ids"], attention_mask=tok["attention_mask"]).pooler_output)
        imgs.append(iv.float().cpu().numpy())
        txts.append(tv.float().cpu().numpy())
        print(f"{split}: {min(i + batch, len(rows))}/{len(rows)}", flush=True)

    img_arr = np.concatenate(imgs, axis=0)
    txt_arr = np.concatenate(txts, axis=0)
    ids_arr = np.array([r["id"] for r in rows])
    lab_arr = np.array([r["label"] for r in rows], dtype=np.int64)
    return img_arr, txt_arr, ids_arr, lab_arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=SPLITS)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading CLIP {MODEL} on {device}...")
    model = CLIPModel.from_pretrained(MODEL).to(device).eval()
    proc = CLIPProcessor.from_pretrained(MODEL)
    os.makedirs(a.out, exist_ok=True)
    for s in a.splits:
        target_file = os.path.join(a.out, f"{s}_img.npy")
        if os.path.exists(target_file):
            print(f"Skipping {s} (already cached at {target_file})")
            continue
        img, txt, ids, lab = extract_split(s, model, proc, device, a.batch)
        for name, arr in [("img", img), ("txt", txt), ("ids", ids), ("labels", lab)]:
            np.save(os.path.join(a.out, f"{s}_{name}.npy"), arr)
        print(f"Successfully cached {s}: img={img.shape}, txt={txt.shape}")


def check(out=OUT_DIR, data_dir=DATA_DIR):
    """Verify shapes, row counts, and alignment against raw splits."""
    for s in SPLITS:
        rows = load_jsonl(s, data_dir)
        img = np.load(os.path.join(out, f"{s}_img.npy"))
        txt = np.load(os.path.join(out, f"{s}_txt.npy"))
        ids = np.load(os.path.join(out, f"{s}_ids.npy"))
        lab = np.load(os.path.join(out, f"{s}_labels.npy"))
        assert img.shape == txt.shape == (len(rows), DIM), f"Shape mismatch in {s}: {img.shape} vs {len(rows)}"
        assert list(ids) == [r["id"] for r in rows], f"ID mismatch in {s}"
        assert list(lab) == [r["label"] for r in rows], f"Label mismatch in {s}"
        assert np.isfinite(img).all() and np.isfinite(txt).all(), f"Non-finite values in {s}"
        print(f"OK {s}: {img.shape}")


if __name__ == "__main__":
    main()
    check()
