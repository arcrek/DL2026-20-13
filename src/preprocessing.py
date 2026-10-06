"""Data preprocessing pipeline for Memotion 7k.

Implements the standard preprocessing workflow:
- 3.1 Create clean DataFrame and normalize sentiment labels (5-scale to 3 classes).
- 3.2 Filter corrupted/truncated images before training to prevent 'OSError: image file is truncated'.
- 3.3 Encode string labels to numeric IDs (negative: 0, neutral: 1, positive: 2).
"""
import os
from typing import Callable, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from PIL import Image

# 3.3 Numeric label mappings
LABEL2ID: Dict[str, int] = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

ID2LABEL: Dict[int, str] = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

DEFAULT_DATA_DIR = os.environ.get("DATA_DIR", "data/memotion")


def normalize_sentiment(label: Optional[Union[str, float]]) -> Optional[str]:
    """Normalize raw sentiment labels into 3 target classes: positive, neutral, negative.

    Memotion 7k raw labels may include:
    very_positive, positive, neutral, negative, very_negative.

    Consolidated mapping:
    - 'very positive', 'positive', 'pos' -> 'positive'
    - 'neutral', 'neu'                   -> 'neutral'
    - 'very negative', 'negative', 'neg' -> 'negative'
    """
    if pd.isna(label):
        return None

    label = str(label).strip().lower()
    label = label.replace("_", " ")

    if label in ["very positive", "positive", "pos"]:
        return "positive"
    if label in ["neutral", "neu"]:
        return "neutral"
    if label in ["very negative", "negative", "neg"]:
        return "negative"

    return None


def get_image_path(
    image_name: Optional[Union[str, float]],
    base_dir: Optional[str] = None,
    search_dirs: Optional[List[str]] = None,
) -> Optional[str]:
    """Resolve an absolute or valid relative path for a given image file name.

    Returns the existing file path if found on disk, otherwise returns None.
    """
    if pd.isna(image_name) or not image_name:
        return None

    img_str = str(image_name).strip()
    if not img_str:
        return None

    # Check if already a valid path
    if os.path.exists(img_str):
        return os.path.abspath(img_str)

    dirs_to_check: List[str] = []
    if base_dir:
        dirs_to_check.append(base_dir)
        dirs_to_check.append(os.path.join(base_dir, "img"))
        dirs_to_check.append(os.path.join(base_dir, "images"))

    if search_dirs:
        dirs_to_check.extend(search_dirs)

    # Standard locations
    dirs_to_check.extend([
        os.path.join(DEFAULT_DATA_DIR, "img"),
        DEFAULT_DATA_DIR,
        "/kaggle/input/datasets/williamscott701/memotion-dataset-7k/images",
        "/kaggle/input/datasets/williamscott701/memotion-dataset-7k/memotion_dataset_7k/images",
        "data/memotion/img",
        "data/memotion",
    ])

    # Possible extensions to try if extension is omitted
    candidates = [img_str]
    root, ext = os.path.splitext(img_str)
    if not ext:
        candidates.extend([f"{root}.png", f"{root}.jpg", f"{root}.jpeg"])

    for d in dirs_to_check:
        for c in candidates:
            p = os.path.join(d, c)
            if os.path.exists(p):
                return os.path.abspath(p)

    return None


def create_clean_dataframe(
    df: pd.DataFrame,
    image_dir: Optional[str] = None,
    get_image_path_fn: Optional[Callable[[str], Optional[str]]] = None,
    text_col: str = "text_ocr",
    label_col: str = "overall_sentiment",
    image_col: str = "image_name",
    verbose: bool = True,
) -> pd.DataFrame:
    """3.1 Create clean DataFrame for training.

    After this step, the dataset contains only 3 primary columns:
    - text
    - image_path
    - label_name

    Samples missing valid image paths or sentiment labels are dropped.
    """
    # Resolve column names with graceful fallbacks
    resolved_text_col = text_col
    if resolved_text_col not in df.columns:
        for candidate in ["text_corrected", "text", "caption"]:
            if candidate in df.columns:
                resolved_text_col = candidate
                break

    resolved_label_col = label_col
    if resolved_label_col not in df.columns:
        for candidate in ["sentiment", "raw_sentiment", "label"]:
            if candidate in df.columns:
                resolved_label_col = candidate
                break

    resolved_image_col = image_col
    if resolved_image_col not in df.columns:
        for candidate in ["img", "image", "image_path", "file_name", "id"]:
            if candidate in df.columns:
                resolved_image_col = candidate
                break

    # Build image path resolver
    if get_image_path_fn is None:
        def _resolver(name):
            return get_image_path(name, base_dir=image_dir)
        get_image_path_fn = _resolver

    data = pd.DataFrame()
    data["text"] = df[resolved_text_col].fillna("").astype(str)
    data["image_path"] = df[resolved_image_col].apply(get_image_path_fn)
    data["label_name"] = df[resolved_label_col].apply(normalize_sentiment)

    if verbose:
        print("Before filtering invalid samples:", len(data))
        print("Samples with found images:", data["image_path"].notna().sum())
        print("Samples with valid labels:", data["label_name"].notna().sum())

    # Drop samples missing image or label
    data = data.dropna(subset=["image_path", "label_name"]).reset_index(drop=True)

    if verbose:
        print("After dropping missing image/label:", len(data))
        print(data["label_name"].value_counts())

    return data


def is_valid_image(path: Union[str, os.PathLike, None]) -> bool:
    """3.2 Filter corrupted/truncated images before training.

    Verifies image file integrity via Image.verify() and Image.convert("RGB").
    Prevents 'OSError: image file is truncated' in PyTorch DataLoader.
    """
    if not path or pd.isna(path):
        return False
    path_str = str(path)
    if not os.path.exists(path_str):
        return False
    try:
        with Image.open(path_str) as img:
            img.verify()

        with Image.open(path_str) as img:
            img.convert("RGB")

        return True
    except Exception:
        return False


def filter_valid_images(
    data: pd.DataFrame,
    image_col: str = "image_path",
    verbose: bool = True,
) -> pd.DataFrame:
    """Filter out rows containing corrupted or truncated images."""
    before_filter = len(data)
    data = data[data[image_col].apply(is_valid_image)].reset_index(drop=True)
    after_filter = len(data)

    if verbose:
        print("Corrupted images dropped:", before_filter - after_filter)
        print("Final sample count for training:", after_filter)
        print(data["label_name"].value_counts())

    return data


def encode_labels(
    data: pd.DataFrame,
    label_col: str = "label_name",
    label2id: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """3.3 Encode string labels to numeric IDs.

    PyTorch models require numeric target labels rather than strings:
    - negative: 0
    - neutral: 1
    - positive: 2
    """
    if label2id is None:
        label2id = LABEL2ID

    data["label"] = data[label_col].map(label2id)
    return data


def preprocess_memotion(
    df: pd.DataFrame,
    image_dir: Optional[str] = None,
    get_image_path_fn: Optional[Callable[[str], Optional[str]]] = None,
    text_col: str = "text_ocr",
    label_col: str = "overall_sentiment",
    image_col: str = "image_name",
    verbose: bool = True,
) -> pd.DataFrame:
    """Execute the complete 3-step preprocessing pipeline:
    - 3.1 Normalize sentiment and create clean DataFrame (3 columns: text, image_path, label_name).
    - 3.2 Filter corrupted/truncated images (verify & RGB conversion).
    - 3.3 Encode labels to numeric IDs (0, 1, 2).
    """
    data = create_clean_dataframe(
        df=df,
        image_dir=image_dir,
        get_image_path_fn=get_image_path_fn,
        text_col=text_col,
        label_col=label_col,
        image_col=image_col,
        verbose=verbose,
    )
    data = filter_valid_images(data, image_col="image_path", verbose=verbose)
    data = encode_labels(data, label_col="label_name")
    return data


def load_raw_memotion_df(
    data_dir: str = DEFAULT_DATA_DIR,
    splits: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load raw dataset records into a pandas DataFrame matching raw Kaggle/SemEval schema."""
    import json
    if splits is None:
        splits = ["train", "validation", "test"]

    records = []
    for s in splits:
        p = os.path.join(data_dir, f"{s}.jsonl")
        if not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    records.append({
                        "image_name": os.path.basename(r.get("img", f"{r['id']}.png")),
                        "text_ocr": r.get("text", ""),
                        "text_corrected": r.get("text", ""),
                        "overall_sentiment": r.get("raw_sentiment", r.get("label")),
                        "split": s,
                        "id": r.get("id"),
                    })
    return pd.DataFrame(records)


if __name__ == "__main__":
    print("Testing data preprocessing pipeline on local Memotion dataset...")
    df_raw = load_raw_memotion_df()
    print(f"Loaded {len(df_raw)} raw records across splits.")
    img_dir = os.path.join(DEFAULT_DATA_DIR, "img")
    clean_df = preprocess_memotion(
        df=df_raw,
        image_dir=img_dir,
        text_col="text_corrected",
        label_col="overall_sentiment",
        image_col="image_name",
        verbose=True,
    )
    print(f"Preprocessing completed successfully: {len(clean_df)} clean samples retained.")
    print(f"Class distribution: {dict(clean_df['label_name'].value_counts())}")

