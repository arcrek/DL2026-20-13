#!/usr/bin/env bash
# Download Memotion 7k dataset (SemEval-2020 Task 8) from Hugging Face (Ahren09/MMSoc_Memotion)
# Usage: bash scripts/download_data.sh [--jsonl-only]
set -euo pipefail
OUT="${DATA_DIR:-data/memotion}"
mkdir -p "$OUT/img"

# Ensure huggingface_hub, pyarrow, datasets, pillow are installed
if ! python3 -c "import huggingface_hub, pyarrow, PIL" 2>/dev/null; then
  if [ -z "${VIRTUAL_ENV:-}" ] && [ ! -d /content ]; then
    [ -d .venv ] || python3 -m venv .venv
    . .venv/bin/activate
  fi
  pip install -q huggingface_hub pyarrow pillow
fi

python3 - "$OUT" "${1:-}" <<'PY'
import json
import os
import sys
import urllib.request
import io
from PIL import Image

out_dir = sys.argv[1]
flag = sys.argv[2] if len(sys.argv) > 2 else ""

try:
    import pyarrow.parquet as pq
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pyarrow"])
    import pyarrow.parquet as pq

from huggingface_hub import hf_hub_download

repo_id = "Ahren09/MMSoc_Memotion"
splits_files = {
    "train": ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"],
    "validation": ["data/validation-00000-of-00001.parquet"],
    "test": ["data/test-00000-of-00001.parquet"]
}

SENTIMENT_MAP = {
    "very_negative": 0, "negative": 0,
    "neutral": 1,
    "positive": 2, "very_positive": 2
}

token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None

for split, files in splits_files.items():
    jsonl_path = os.path.join(out_dir, f"{split}.jsonl")
    if os.path.exists(jsonl_path) and os.path.getsize(jsonl_path) > 100:
        print(f"{split}.jsonl already exists, skipping download...")
        continue
    
    print(f"Downloading {split} split files...")
    rows = []
    idx = 0
    for f in files:
        local_parquet = hf_hub_download(repo_id, filename=f, repo_type="dataset", token=token)
        table = pq.read_table(local_parquet)
        pydict = table.to_pydict()
        n_rows = len(pydict["text_corrected"])
        
        for i in range(n_rows):
            sample_id = f"{split}_{idx:05d}"
            idx += 1
            raw_sent = str(pydict["sentiment"][i]).strip().lower()
            label = SENTIMENT_MAP.get(raw_sent, 1) # default neutral if unknown
            text = str(pydict["text_corrected"][i] or "").strip()
            img_rel_path = f"img/{sample_id}.png"
            img_abs_path = os.path.join(out_dir, img_rel_path)
            
            if flag != "--jsonl-only" and not os.path.exists(img_abs_path):
                img_bytes = pydict["image"][i].get("bytes")
                if img_bytes:
                    with open(img_abs_path, "wb") as img_f:
                        img_f.write(img_bytes)
                else:
                    # fallback if stored as path
                    pass
            
            row = {
                "id": sample_id,
                "img": img_rel_path,
                "text": text,
                "label": label,
                "raw_sentiment": raw_sent,
                "humor": str(pydict["humor"][i] or ""),
                "sarcasm": str(pydict["sarcasm"][i] or ""),
                "motivational": str(pydict["motivational"][i] or ""),
            }
            rows.append(row)
            
    with open(jsonl_path, "w", encoding="utf-8") as out_f:
        for r in rows:
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} samples to {jsonl_path}")

print("Dataset ready in", out_dir)
PY
