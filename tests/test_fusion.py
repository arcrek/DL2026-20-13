"""Unit tests for multimodal fusion architectures."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from models.fusion import (  # noqa: E402
    ConcatFusionModel,
    CrossAttentionFusionModel,
    ProductFusionModel,
    build_fusion_model,
)


def test_build_fusion_model():
    m_concat = build_fusion_model("both_concat")
    assert isinstance(m_concat, ConcatFusionModel)

    m_cross = build_fusion_model("both_cross_attn")
    assert isinstance(m_cross, CrossAttentionFusionModel)

    m_prod = build_fusion_model("both_product")
    assert isinstance(m_prod, ProductFusionModel)


def test_fusion_forward_shapes():
    batch_size = 2
    seq_len = 16
    pixel_values = torch.randn(batch_size, 3, 224, 224)
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    for config in ["both_concat", "both_cross_attn", "both_product"]:
        model = build_fusion_model(config, freeze_image=True)
        model.eval()
        with torch.no_grad():
            logits = model(pixel_values, input_ids, attention_mask)
        assert logits.shape == (batch_size, 3), f"Expected shape ({batch_size}, 3), got {logits.shape}"
        assert len(model.backbone_params()) > 0
        assert len(model.head_params()) > 0


if __name__ == "__main__":
    test_build_fusion_model()
    test_fusion_forward_shapes()
    print("ALL FUSION TESTS PASSED")
