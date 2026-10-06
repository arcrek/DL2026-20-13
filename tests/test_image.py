import os
import sys
import tempfile
from PIL import Image
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from models.image import ImageArgs, ResizeAndPad, valid_image


def test_resize_and_pad():
    transform = ResizeAndPad()
    # Test on wide image
    img_wide = Image.new("RGB", (400, 200), color=(255, 0, 0))
    out_wide = transform(img_wide)
    assert out_wide.size == (224, 224)
    assert out_wide.mode == "RGB"

    # Test on tall image
    img_tall = Image.new("RGB", (100, 300), color=(0, 255, 0))
    out_tall = transform(img_tall)
    assert out_tall.size == (224, 224)
    assert out_tall.mode == "RGB"

    # Test on square image
    img_sq = Image.new("RGB", (150, 150), color=(0, 0, 255))
    out_sq = transform(img_sq)
    assert out_sq.size == (224, 224)
    assert out_sq.mode == "RGB"


def test_valid_image():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create valid image
        valid_path = os.path.join(tmp_dir, "valid.png")
        Image.new("RGB", (64, 64), color=(100, 100, 100)).save(valid_path)
        assert valid_image(valid_path) is True

        # Create corrupted image file
        corrupt_path = os.path.join(tmp_dir, "corrupt.png")
        with open(corrupt_path, "wb") as f:
            f.write(b"not an actual image file")
        assert valid_image(corrupt_path) is False

        # Non-existent file
        assert valid_image(os.path.join(tmp_dir, "nonexistent.png")) is False


def test_image_args_defaults():
    args = ImageArgs()
    assert args.seeds == [0, 1, 2]
    assert args.batch_size == 32
    assert args.head_epochs == 3
    assert args.finetune_epochs == 5
    assert args.workers == 2
    assert "memotion" in args.data_dir


def test_image_args_custom():
    args = ImageArgs(seeds=[0], batch_size=16, head_epochs=1, finetune_epochs=2, workers=0)
    assert args.seeds == [0]
    assert args.batch_size == 16
    assert args.head_epochs == 1
    assert args.finetune_epochs == 2
    assert args.workers == 0
