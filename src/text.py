"""CLI runner for unimodal text baseline (delegates to src.models.text)."""
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from models.text import (  # noqa: F401
    BertMemeClassifier,
    TextMemeDataset,
    evaluate_split,
    main,
    make_collate_fn,
    train_seed,
)

if __name__ == "__main__":
    main()
