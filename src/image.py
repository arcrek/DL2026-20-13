import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from models.image import (  # noqa: F401
    ImageArgs,
    ImageModel,
    MemeDataset,
    ResizeAndPad,
    evaluate,
    main,
    run_training,
    train_epoch,
    train_seed,
    valid_image,
)

if __name__ == "__main__":
    main()
