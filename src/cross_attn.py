import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from models.cross_attention import (  # noqa: F401
    CONFIG_NAME,
    NUM_CLASSES,
    TEXT_MODEL_NAME,
    CrossAttentionArgs,
    CrossAttentionFusionModel,
    FusionCollate,
    MultimodalMemeDataset,
    ResNet50Backbone,
    build_loss_and_optimizer,
    evaluate,
    evaluate_split,
    main,
    make_multimodal_collate_fn,
    run_training,
    safe_open_image,
    train_epoch,
    train_loop,
    train_seed,
)

if __name__ == "__main__":
    main()
