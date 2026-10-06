"""Tests for confusion matrix calculations and outputs."""
import os
import sys
import tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from make_confusion_matrix import compute_cm_stats, load_model_results, CLASS_NAMES


def test_compute_cm_stats_shapes_and_values():
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 2, 2, 0]
    raw_cm, norm_cm = compute_cm_stats(y_true, y_pred)

    assert raw_cm.shape == (3, 3)
    assert norm_cm.shape == (3, 3)
    assert raw_cm.sum() == 6
    # Each row sum of norm_cm should be 1.0 (for non-empty rows)
    np.testing.assert_allclose(norm_cm.sum(axis=1), np.ones(3))
    # True class 0: one predicted 0, one predicted 1
    assert raw_cm[0, 0] == 1
    assert raw_cm[0, 1] == 1
    assert norm_cm[0, 0] == 0.5


def test_compute_cm_stats_zero_row():
    # If a class has 0 true samples, normalized row should be 0 without crashing
    y_true = [1, 2]
    y_pred = [1, 2]
    raw_cm, norm_cm = compute_cm_stats(y_true, y_pred)
    assert raw_cm[0].sum() == 0
    assert norm_cm[0].sum() == 0.0


def test_load_model_results():
    # Verify loading on actual results directory
    res = load_model_results("both_concat", seeds=[0, 1, 2], results_dir="results")
    assert len(res["y_true"]) == 2100  # 700 * 3
    assert len(res["y_pred"]) == 2100
    assert len(res["seed_cms"]) == 3
    assert len(res["seed_accuracies"]) == 3
    for cm in res["seed_cms"]:
        assert cm.shape == (3, 3)
        assert cm.sum() == 700
