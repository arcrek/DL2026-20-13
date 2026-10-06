import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from models.text import (  # noqa: F401
    MODEL_NAME,
    NUM_CLASSES,
    BertMemeClassifier,
    TextMemeDataset,
    build_loss_and_optimizer,
    evaluate,
    evaluate_split,
    main,
    make_collate_fn,
    train_epoch,
    train_loop,
    train_seed,
)

if __name__ == "__main__":
    main()
