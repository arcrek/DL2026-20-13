"""Loader for the shared experiment config (configs/base.yaml)."""
import os

import yaml

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "base.yaml")


def load_config(path=DEFAULT_PATH):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg["seeds"] != [0, 1, 2]:
        raise ValueError("seeds must be [0, 1, 2] for every model")
    return cfg
