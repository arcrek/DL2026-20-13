"""CLI runner for multimodal fusion models (delegates to src.models.fusion)."""
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from models.fusion import (  # noqa: F401
    ConcatFusionModel,
    CrossAttentionFusionModel,
    MultimodalMemeDataset,
    ProductFusionModel,
    build_fusion_model,
    evaluate_split,
    main,
    make_fusion_collate_fn,
    train_seed,
)

if __name__ == "__main__":
    main()
