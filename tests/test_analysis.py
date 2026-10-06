"""Tests for pairing, score computation, and the statistical unit of resampling."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis import (MODELS, Experiment, holm_adjust, load_experiment, load_metadata,
                          metrics, modality_complementarity, paired_bootstrap, paired_permutation)
from src.results import save_results


def test_confusion_scores_match_independent_sklearn_implementation():
    rng = np.random.default_rng(7)
    y = rng.integers(0, 3, 80)
    predictions = rng.integers(0, 3, (3, 80))
    actual_f1, actual_acc, actual_per_class = metrics(y, predictions)
    for seed, pred in enumerate(predictions):
        assert actual_f1[seed] == pytest.approx(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0))
        assert actual_acc[seed] == pytest.approx(accuracy_score(y, pred))
        assert actual_per_class[seed] == pytest.approx(f1_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0))


def test_bootstrap_keeps_paired_models_and_is_deterministic():
    y = np.array([0, 1, 2, 0, 1, 2])
    pred = np.stack([y, (y + 1) % 3, y])
    predictions = {"a": pred, "b": pred.copy()}
    first = paired_bootstrap(y, predictions, 75, 15)
    second = paired_bootstrap(y, predictions, 75, 15)
    assert np.array_equal(first, second)
    assert np.array_equal(first[:, 0], first[:, 1])
    assert first[:, 0] == pytest.approx(np.full(75, 2 / 3))


def test_exact_permutation_uses_memes_as_blocks_not_seed_replicates():
    y = np.array([0, 1, 2, 0])
    a, b = np.tile(y, (3, 1)), np.tile((y + 1) % 3, (3, 1))
    result = paired_permutation(y, a, b)
    assert result["exact"]
    assert result["draws"] == 16  # Four meme blocks, not twelve independent predictions.
    assert result["delta_macro_f1"] == pytest.approx(1)
    assert result["p_raw"] == pytest.approx(2 / 16)
    reverse = paired_permutation(y, b, a)
    assert reverse["p_raw"] == result["p_raw"]
    assert reverse["delta_macro_f1"] == -result["delta_macro_f1"]
    assert paired_permutation(y, a, a)["p_raw"] == 1


def test_holm_adjustment_is_monotone_in_ordered_pvalues():
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm_adjust([0.6, 0.8]) == [1, 1]


def test_modality_complementarity_partitions_the_test_set_without_pooling_sample_counts():
    y = np.array([0, 1, 2, 0])
    predictions = {"text": np.tile([0, 0, 2, 1], (3, 1)),
                   "image": np.tile([0, 1, 0, 1], (3, 1)),
                   **{m: np.tile(y, (3, 1)) for m in MODELS[2:]}}
    experiment = Experiment([str(i) for i in range(4)], y, np.array(["general"]*4),
                            predictions, {}, [], {})
    rows = modality_complementarity(experiment)
    assert {r["state"] for r in rows} == {"text_only_correct", "image_only_correct", "both_correct", "both_wrong"}
    assert sum(r["n_mean_per_seed"] for r in rows) == 4
    assert all(r["n_mean_per_seed"] == 1 and r["n_seeds"] == [1, 1, 1] for r in rows)
    assert all(r["both_concat_correct_rate"] == 1 for r in rows)


def make_results(path):
    ids = [f"test_{i:05d}" for i in range(6)]
    y = np.array([0, 1, 2, 0, 1, 2])
    probs = np.eye(3)[y] * 0.8 + 0.2 / 3
    sarcasm = ["general", "not_sarcastic"] * 3
    for model in MODELS:
        for seed in (0, 1, 2):
            # Force different row order in image files; alignment must use IDs.
            order = np.arange(6)[::-1] if model == "image" else np.arange(6)
            save_results(model, seed, [ids[i] for i in order], y[order], probs[order],
                         [sarcasm[i] for i in order], results_dir=path)
    return ids, y, sarcasm


def test_loader_aligns_ids_and_rejects_incompatible_gold_labels(tmp_path):
    ids, y, _ = make_results(tmp_path)
    experiment = load_experiment(tmp_path)
    assert experiment.ids == ids
    assert np.array_equal(experiment.predictions["text"], experiment.predictions["image"])
    path = tmp_path / "image_seed1.json"
    result = json.loads(path.read_text())
    result["y_true"][0] = (result["y_true"][0] + 1) % 3
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="Labels or sarcasm metadata differ"):
        load_experiment(tmp_path)


def test_metadata_labels_must_match_saved_predictions(tmp_path):
    results = tmp_path / "results"
    ids, y, sarcasm = make_results(results)
    experiment = load_experiment(results)
    rows = [{"id": i, "label": int(label), "sarcasm": sar, "humor": "not_funny"}
            for i, label, sar in zip(ids, y, sarcasm)]
    path = tmp_path / "metadata.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows[::-1]))
    assert [r["id"] for r in load_metadata(path, experiment)] == ids
    rows[0]["sarcasm"] = "not_sarcastic"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match="Dataset annotation mismatch"):
        load_metadata(path, experiment)


@pytest.mark.parametrize("fault,match", [("negative_probability", "outside"),
                                        ("wrong_prediction", "argmax"),
                                        ("stale_metric", "Stored Macro-F1")])
def test_loader_rejects_invalid_probabilities_predictions_and_stale_scores(tmp_path, fault, match):
    make_results(tmp_path)
    path = tmp_path / "both_concat_seed2.json"
    result = json.loads(path.read_text())
    if fault == "negative_probability":
        result["probs"][0] = [-0.1, 1.1, 0.0]
    elif fault == "wrong_prediction":
        result["y_pred"][0] = 1
    else:
        result["macro_f1"] = 0.5
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match=match):
        load_experiment(tmp_path)
