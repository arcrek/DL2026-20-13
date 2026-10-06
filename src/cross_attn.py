import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import models.cross_attention as _mca

# Re-export cross-attention components with fallback resilience
CONFIG_NAME = getattr(_mca, "CONFIG_NAME", "both_cross_attn")
NUM_CLASSES = getattr(_mca, "NUM_CLASSES", 3)
TEXT_MODEL_NAME = getattr(_mca, "TEXT_MODEL_NAME", "bert-base-uncased")
CrossAttentionArgs = getattr(_mca, "CrossAttentionArgs", None)
CrossAttentionFusionModel = getattr(_mca, "CrossAttentionFusionModel", None)
FusionCollate = getattr(_mca, "FusionCollate", None)
MultimodalMemeDataset = getattr(_mca, "MultimodalMemeDataset", None)
ResNet50Backbone = getattr(_mca, "ResNet50Backbone", None)
build_loss_and_optimizer = getattr(_mca, "build_loss_and_optimizer", None)
evaluate = getattr(_mca, "evaluate", None)
evaluate_split = getattr(_mca, "evaluate_split", evaluate)
main = getattr(_mca, "main", None)
make_multimodal_collate_fn = getattr(_mca, "make_multimodal_collate_fn", None)
run_training = getattr(_mca, "run_training", None)
safe_open_image = getattr(_mca, "safe_open_image", None)
train_epoch = getattr(_mca, "train_epoch", None)
train_loop = getattr(_mca, "train_loop", None)
train_seed = getattr(_mca, "train_seed", train_loop)


__all__ = [
    "CONFIG_NAME",
    "NUM_CLASSES",
    "TEXT_MODEL_NAME",
    "CrossAttentionArgs",
    "CrossAttentionFusionModel",
    "FusionCollate",
    "MultimodalMemeDataset",
    "ResNet50Backbone",
    "build_loss_and_optimizer",
    "evaluate",
    "evaluate_split",
    "main",
    "make_multimodal_collate_fn",
    "run_training",
    "safe_open_image",
    "train_epoch",
    "train_loop",
    "train_seed",
]

if __name__ == "__main__":
    main()

