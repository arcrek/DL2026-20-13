"""Unit tests for Cross-Attention Multimodal Fusion architecture (Branch 3B)."""
import os
import sys
import tempfile
import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from models.cross_attention import (  # noqa: E402
    CrossAttentionArgs,
    CrossAttentionFusionModel,
    MultimodalMemeDataset,
    ResNet50Backbone,
    make_multimodal_collate_fn,
)
import results  # noqa: E402


def test_resnet50_backbone_shapes():
    bb = ResNet50Backbone(pretrained=False, freeze=True)
    bb.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        pooled, spatial = bb(x)
    assert pooled.shape == (2, 2048)
    assert spatial.shape == (2, 49, 2048)  # 7x7=49 spatial patches


def test_cross_attention_forward_shapes():
    batch_size = 2
    seq_len = 16
    pixel_values = torch.randn(batch_size, 3, 224, 224)
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    model = CrossAttentionFusionModel(
        num_classes=3,
        attn_dim=64,
        num_heads=2,
        freeze_image=True,
    )
    model.eval()

    with torch.no_grad():
        logits = model(pixel_values, input_ids, attention_mask)

    assert logits.shape == (batch_size, 3)
    assert len(model.backbone_params()) > 0
    assert len(model.head_params()) > 0


def test_multimodal_dataset_and_collate():
    from transformers import AutoTokenizer

    rows = [
        {"id": "sample_001", "img": "nonexistent.png", "text": "funny meme text", "label": 2, "sarcasm": "general"},
        {"id": "sample_002", "img": "dummy.png", "text": "neutral post", "label": 1, "sarcasm": "not_sarcastic"},
    ]
    ds = MultimodalMemeDataset(rows)
    assert len(ds) == 2
    item = ds[0]
    assert item["id"] == "sample_001"
    assert item["label"] == 2

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    collate = make_multimodal_collate_fn(tokenizer, max_length=32)
    batch = collate([ds[0], ds[1]])

    assert batch["ids"] == ["sample_001", "sample_002"]
    assert batch["pixel_values"].shape == (2, 3, 224, 224)
    assert batch["input_ids"].shape[0] == 2
    assert batch["labels"].tolist() == [2, 1]
    assert batch["sarcasms"] == ["general", "not_sarcastic"]


def test_cross_attn_results_schema():
    for seed in [0, 1, 2]:
        res = results.load_results("both_cross_attn", seed)
        assert res["config"] == "both_cross_attn"
        assert res["seed"] == seed
        assert res["split"] == "test"
        assert len(res["ids"]) == 700
        assert 0.0 <= res["macro_f1"] <= 1.0
        assert 0.0 <= res["accuracy"] <= 1.0


def test_cross_attention_args_defaults():
    args = CrossAttentionArgs()
    assert args.seeds == [0, 1, 2]
    assert args.epochs == 5
    assert args.batch_size == 16
    assert args.eval_batch_size == 32
    assert args.lr_backbone == 1.5e-5
    assert args.lr_head == 5e-4
    assert args.attn_dim == 256
    assert args.num_heads == 4
    assert args.workers == 2
    assert args.freeze_image is True
    assert "memotion" in args.data_dir


def test_cross_attention_args_custom():
    args = CrossAttentionArgs(seeds=[0], epochs=1, batch_size=8, attn_dim=128, num_heads=2)
    assert args.seeds == [0]
    assert args.epochs == 1
    assert args.batch_size == 8
    assert args.attn_dim == 128
    assert args.num_heads == 2


def test_multimodal_dataset_preprocessed_records():
    rows = [
        {"id": "clean_001", "image_path": "dummy.png", "text": "clean text", "label_name": "positive", "label": 2},
        {"id": "clean_002", "image_path": "dummy2.png", "text": "neutral post", "label_name": "neutral", "label": 1},
    ]
    ds = MultimodalMemeDataset(rows)
    assert len(ds) == 2
    assert ds[0]["img_path"] == "dummy.png"
    assert ds[0]["text"] == "clean text"
    assert ds[0]["label"] == 2


def test_cross_attn_public_api_and_exports():
    import cross_attn
    import models.cross_attention as mca

    assert hasattr(cross_attn, "CrossAttentionArgs")
    assert hasattr(cross_attn, "train_seed")
    assert hasattr(cross_attn, "train_loop")
    assert hasattr(cross_attn, "evaluate_split")
    assert hasattr(cross_attn, "run_training")
    assert callable(cross_attn.train_seed)
    assert cross_attn.train_seed is mca.train_seed
    assert cross_attn.CrossAttentionArgs is mca.CrossAttentionArgs
    assert cross_attn.CrossAttentionFusionModel is mca.CrossAttentionFusionModel


if __name__ == "__main__":
    test_resnet50_backbone_shapes()
    test_cross_attention_forward_shapes()
    test_multimodal_dataset_and_collate()
    test_cross_attn_results_schema()
    test_cross_attention_args_defaults()
    test_cross_attention_args_custom()
    test_multimodal_dataset_preprocessed_records()
    test_cross_attn_public_api_and_exports()
    print("ALL CROSS-ATTENTION TESTS PASSED")

