"""Unit tests for data preprocessing module (src/preprocessing.py)."""
import os
import sys
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocessing import (
    ID2LABEL,
    LABEL2ID,
    create_clean_dataframe,
    encode_labels,
    filter_valid_images,
    get_image_path,
    is_valid_image,
    normalize_sentiment,
    preprocess_memotion,
)


def test_normalize_sentiment():
    # Positive variations
    assert normalize_sentiment("very_positive") == "positive"
    assert normalize_sentiment("very positive") == "positive"
    assert normalize_sentiment("positive") == "positive"
    assert normalize_sentiment("pos") == "positive"
    assert normalize_sentiment("POSITIVE") == "positive"
    assert normalize_sentiment(" Very_Positive ") == "positive"

    # Neutral variations
    assert normalize_sentiment("neutral") == "neutral"
    assert normalize_sentiment("neu") == "neutral"
    assert normalize_sentiment("NEUTRAL") == "neutral"

    # Negative variations
    assert normalize_sentiment("very_negative") == "negative"
    assert normalize_sentiment("very negative") == "negative"
    assert normalize_sentiment("negative") == "negative"
    assert normalize_sentiment("neg") == "negative"

    # Invalid / Missing
    assert normalize_sentiment(None) is None
    assert normalize_sentiment(float("nan")) is None
    assert normalize_sentiment("") is None
    assert normalize_sentiment("unknown_label") is None


def test_label_mappings():
    assert LABEL2ID == {"negative": 0, "neutral": 1, "positive": 2}
    assert ID2LABEL == {0: "negative", 1: "neutral", 2: "positive"}
    for k, v in LABEL2ID.items():
        assert ID2LABEL[v] == k


def test_is_valid_image():
    # Valid existing image
    valid_path = "data/memotion/img/train_00000.png"
    if os.path.exists(valid_path):
        assert is_valid_image(valid_path) is True

    # Truncated corrupted image (train_04578.png is known truncated)
    corrupted_path = "data/memotion/img/train_04578.png"
    if os.path.exists(corrupted_path):
        assert is_valid_image(corrupted_path) is False

    # Nonexistent path
    assert is_valid_image("nonexistent_image.png") is False
    assert is_valid_image(None) is False


def test_create_clean_dataframe_and_filter():
    # Sample mock df
    df = pd.DataFrame({
        "image_name": ["train_00000.png", "missing.png", "train_04578.png", "train_00001.png"],
        "text_ocr": ["Meme caption 1", "Meme caption 2", "Meme caption 3", None],
        "overall_sentiment": ["very_positive", "neutral", "positive", "invalid_sentiment"],
    })

    # Step 10: Clean dataframe
    clean_df = create_clean_dataframe(df, image_dir="data/memotion/img", verbose=False)

    # "missing.png" has no image_path -> dropped
    # "train_00001.png" has invalid_sentiment -> dropped
    # Remaining: train_00000.png and train_04578.png
    assert len(clean_df) == 2
    assert list(clean_df.columns) == ["text", "image_path", "label_name"]
    assert clean_df.iloc[0]["label_name"] == "positive"
    assert clean_df.iloc[1]["label_name"] == "positive"

    # Step 11: Filter invalid image (train_04578 is corrupt)
    filtered_df = filter_valid_images(clean_df, verbose=False)
    assert len(filtered_df) == 1
    assert "train_00000.png" in filtered_df.iloc[0]["image_path"]

    # Step 12: Encode labels to numeric
    encoded_df = encode_labels(filtered_df)
    assert "label" in encoded_df.columns
    assert encoded_df.iloc[0]["label"] == 2


def test_full_pipeline_on_synthetic_data(tmp_path):
    from PIL import Image
    # Create temporary images: 1 valid, 1 truncated/corrupt
    valid_img_path = tmp_path / "img_valid.png"
    Image.new("RGB", (10, 10), color="red").save(valid_img_path)

    corrupt_img_path = tmp_path / "img_corrupt.png"
    with open(corrupt_img_path, "wb") as f:
        # Partial PNG header without complete image data
        f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01")

    df = pd.DataFrame({
        "image_name": ["img_valid.png", "img_corrupt.png", "img_nonexistent.png"],
        "text_ocr": ["Valid meme", "Corrupt meme", "Missing meme"],
        "overall_sentiment": ["pos", "neg", "neu"],
    })

    processed = preprocess_memotion(
        df=df,
        image_dir=str(tmp_path),
        verbose=False,
    )

    assert len(processed) == 1
    assert processed.iloc[0]["text"] == "Valid meme"
    assert processed.iloc[0]["label_name"] == "positive"
    assert processed.iloc[0]["label"] == 2
