"""Reproducible Role E analysis of saved, paired Memotion predictions.

Run from the repository root: python -m src.analysis --report-date 2026-10-07
No model training, checkpoint loading, or network access is performed here.
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import csv
import hashlib
import itertools
import json
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .config import load_config
    from .results import validate_result
except ImportError:
    from config import load_config
    from results import validate_result


MODELS = ("text", "image", "both_concat", "both_cross_attn", "both_product")
NAMES = {"text": "BERT", "image": "ResNet50", "both_concat": "Concat",
         "both_cross_attn": "Cross-Attention", "both_product": "Product"}
CLASSES = ("Negative", "Neutral", "Positive")
SARCASTIC = {"general", "twisted_meaning", "very_twisted"}
SAR_LABELS = SARCASTIC | {"not_sarcastic"}
HUMOR_LABELS = {"not_funny", "funny", "very_funny", "hilarious"}
COLORS = ("#3568a8", "#8b6bb1", "#159588", "#e39339", "#cc6375")


@dataclass
class Experiment:
    ids: list
    y: np.ndarray
    sarcasm: np.ndarray
    predictions: dict
    probabilities: dict
    runs: list
    fingerprints: dict
    seeds: tuple = (0, 1, 2)


def confusion(y, predictions):
    """Confusion counts with shape (seed, actual class, predicted class)."""
    predictions = np.atleast_2d(predictions)
    offsets = 9 * np.arange(len(predictions))[:, None]
    encoded = offsets + 3 * np.asarray(y)[None, :] + predictions
    return np.bincount(encoded.ravel(), minlength=9 * len(predictions)).reshape(-1, 3, 3)


def scores_from_confusion(cm):
    cm = np.asarray(cm, dtype=float)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    denominator = cm.sum(-1) + cm.sum(-2)
    class_f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    total = cm.sum(axis=(-1, -2))
    accuracy = np.divide(tp.sum(-1), total, out=np.zeros_like(total), where=total > 0)
    return class_f1.mean(-1), accuracy, class_f1


def metrics(y, predictions):
    return scores_from_confusion(confusion(y, predictions))


def load_experiment(results_dir, config_path=None):
    cfg = load_config(config_path) if config_path else load_config()
    if set(cfg["configs"]) != set(MODELS) or cfg["data"]["results_split"] != "test":
        raise ValueError("Expected five model configs evaluated on the test split")
    predictions, probabilities, fingerprints, runs = {}, {}, {}, []
    ids = y = sarcasm = None
    for model in MODELS:
        model_preds, model_probs = [], []
        for seed in cfg["seeds"]:
            path = Path(results_dir) / f"{model}_seed{seed}.json"
            raw = path.read_bytes()
            result = validate_result(json.loads(raw))
            if result["config"] != model or result["seed"] != seed or result["split"] != "test":
                raise ValueError(f"Filename/config/seed/split mismatch: {path}")
            if not result["ids"]:
                raise ValueError(f"Empty test results: {path}")
            if ids is None:
                ids = result["ids"]
            if set(result["ids"]) != set(ids):
                raise ValueError(f"Different sample ID sets: {path}")
            lookup = {sample_id: i for i, sample_id in enumerate(result["ids"])}
            order = [lookup[sample_id] for sample_id in ids]
            current_y = np.asarray(result["y_true"])[order]
            pred = np.asarray(result["y_pred"])[order]
            if current_y.dtype.kind not in "iu" or pred.dtype.kind not in "iu":
                raise ValueError(f"Class labels must be integers: {path}")
            probs = np.asarray(result["probs"], dtype=float)[order]
            if np.any(probs < 0) or np.any(probs > 1):
                raise ValueError(f"Probabilities outside [0,1]: {path}")
            if not np.array_equal(pred, probs.argmax(1)):
                raise ValueError(f"y_pred does not match argmax(probs): {path}")
            current_sarcasm = np.asarray(result["sarcasm"])[order]
            if not set(current_sarcasm).issubset(SAR_LABELS):
                raise ValueError(f"Unknown sarcasm labels: {path}")
            if y is None:
                y, sarcasm = current_y, current_sarcasm
            if not np.array_equal(y, current_y) or not np.array_equal(sarcasm, current_sarcasm):
                raise ValueError(f"Labels or sarcasm metadata differ after ID alignment: {path}")
            f1, accuracy, _ = metrics(y, pred)
            if not np.isclose(f1[0], result["macro_f1"], atol=1e-10, rtol=0):
                raise ValueError(f"Stored Macro-F1 disagrees with predictions: {path}")
            if not np.isclose(accuracy[0], result["accuracy"], atol=1e-10, rtol=0):
                raise ValueError(f"Stored Accuracy disagrees with predictions: {path}")
            runs.append({"model": model, "seed": seed, "n": len(y),
                         "macro_f1": float(f1[0]), "accuracy": float(accuracy[0]),
                         "holdout_macro_f1": result.get("extra", {}).get("holdout_macro_f1"),
                         "best_epoch": result.get("extra", {}).get("best_epoch")})
            model_preds.append(pred)
            model_probs.append(probs)
            fingerprints[path.name] = hashlib.sha256(raw).hexdigest()
        predictions[model] = np.stack(model_preds)
        probabilities[model] = np.stack(model_probs)
    return Experiment(ids, y, sarcasm, predictions, probabilities, runs, fingerprints,
                      tuple(cfg["seeds"]))


def load_metadata(path, experiment):
    if not path or not Path(path).exists():
        return None
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    mapping = {row["id"]: row for row in rows}
    if len(mapping) != len(rows) or set(mapping) != set(experiment.ids):
        raise ValueError("Metadata IDs do not match the evaluated test samples")
    aligned = [mapping[sample_id] for sample_id in experiment.ids]
    for i, row in enumerate(aligned):
        if row["label"] != experiment.y[i] or row["sarcasm"] != experiment.sarcasm[i]:
            raise ValueError(f"Dataset annotation mismatch for {row['id']}")
        if row["humor"] not in HUMOR_LABELS:
            raise ValueError(f"Unknown humor label for {row['id']}")
    return aligned


def summarize(experiment, mask=None):
    mask = np.ones(len(experiment.y), dtype=bool) if mask is None else mask
    if not np.any(mask):
        raise ValueError("Cannot evaluate an empty subgroup")
    summary = []
    for model in MODELS:
        f1, acc, per_class = metrics(experiment.y[mask], experiment.predictions[model][:, mask])
        summary.append({"model": model, "n": int(mask.sum()),
                        "macro_f1_mean": float(f1.mean()), "macro_f1_sd": float(f1.std(ddof=1)),
                        "accuracy_mean": float(acc.mean()), "accuracy_sd": float(acc.std(ddof=1)),
                        "macro_f1_seeds": f1.tolist(), "accuracy_seeds": acc.tolist(),
                        "class_f1_mean": per_class.mean(0).tolist(),
                        "class_counts": np.bincount(experiment.y[mask], minlength=3).tolist()})
    return summary


def paired_bootstrap(y, predictions, n_resamples=20000, seed=20261007, batch_size=256):
    """Class-stratified meme bootstrap, shared across every model and seed.

    Estimates the mean of seed-wise Macro-F1, not the F1 of pooled predictions.
    Confidence intervals condition on the three supplied trained models.
    """
    y = np.asarray(y)
    if len(y) == 0 or n_resamples < 1:
        raise ValueError("Bootstrap needs samples and a positive resample count")
    stacked = np.stack([np.atleast_2d(predictions[model]) for model in predictions])
    if stacked.shape[-1] != len(y):
        raise ValueError("Prediction and label lengths differ")
    n_models, n_seeds, n = stacked.shape
    code = 3 * y[None, None, :] + stacked
    design = (code.transpose(2, 0, 1)[..., None] == np.arange(9)).astype(np.float32)
    design = design.reshape(n, -1)
    rng = np.random.default_rng(seed)
    output = np.empty((n_resamples, n_models))
    strata = [np.flatnonzero(y == c) for c in range(3) if np.any(y == c)]
    for start in range(0, n_resamples, batch_size):
        size = min(batch_size, n_resamples - start)
        weights = np.zeros((size, n), dtype=np.float32)
        for indices in strata:
            weights[:, indices] = rng.multinomial(len(indices), np.full(len(indices), 1 / len(indices)),
                                                  size=size)
        cm = (weights @ design).reshape(size, n_models, n_seeds, 3, 3)
        output[start:start + size] = scores_from_confusion(cm)[0].mean(-1)
    return output


def paired_permutation(y, a, b, n_resamples=20000, seed=20261007, batch_size=256):
    """Two-sided block randomization: swap all seeds together for each meme.

    Uses exact enumeration for <=12 memes, otherwise the Monte Carlo +1 correction.
    The exchangeability null is conditional on the saved prediction vectors.
    """
    y, a, b = np.asarray(y), np.atleast_2d(a), np.atleast_2d(b)
    if a.shape != b.shape or a.shape[-1] != len(y) or not len(y) or n_resamples < 1:
        raise ValueError("Permutation needs compatible nonempty predictions")
    observed = float(metrics(y, a)[0].mean() - metrics(y, b)[0].mean())
    if np.array_equal(a, b):
        return {"delta_macro_f1": observed, "p_raw": 1.0, "exact": True, "draws": 1}
    ca, cb = confusion(y, a), confusion(y, b)
    acode, bcode = (3 * y + a).T, (3 * y + b).T
    changes = ((bcode[..., None] == np.arange(9)).astype(np.float32)
               - (acode[..., None] == np.arange(9)).astype(np.float32)).reshape(len(y), -1)
    exact = len(y) <= 12
    draws = 2 ** len(y) if exact else n_resamples
    rng = np.random.default_rng(seed)
    extreme = 0
    for start in range(0, draws, batch_size):
        size = min(batch_size, draws - start)
        if exact:
            masks = ((np.arange(start, start + size)[:, None] >> np.arange(len(y))) & 1)
        else:
            masks = rng.integers(0, 2, size=(size, len(y)))
        change = (masks.astype(np.float32) @ changes).reshape(size, a.shape[0], 3, 3)
        null = (scores_from_confusion(ca + change)[0].mean(-1)
                - scores_from_confusion(cb - change)[0].mean(-1))
        extreme += int(np.count_nonzero(np.abs(null) >= abs(observed) - 1e-12))
    p = extreme / draws if exact else (extreme + 1) / (draws + 1)
    return {"delta_macro_f1": observed, "p_raw": float(p), "exact": exact, "draws": draws}


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.maximum.accumulate((len(pvalues) - np.arange(len(pvalues))) * pvalues[order])
    output = np.empty_like(pvalues)
    output[order] = np.minimum(adjusted, 1)
    return output.tolist()


def comparisons(experiment, bootstrap, draws, random_seed):
    primary = [(fusion, base) for fusion in MODELS[2:] for base in MODELS[:2]]
    exploratory = [("text", "image")] + list(itertools.combinations(MODELS[2:], 2))
    records = []
    for i, (a, b) in enumerate(primary + exploratory):
        test = paired_permutation(experiment.y, experiment.predictions[a], experiment.predictions[b],
                                  draws, random_seed + 100 + i)
        distribution = bootstrap[:, MODELS.index(a)] - bootstrap[:, MODELS.index(b)]
        low, high = np.quantile(distribution, [0.025, 0.975])
        records.append({"model_a": a, "model_b": b, "family": "primary" if i < 6 else "exploratory",
                        **test, "ci95_low": float(low), "ci95_high": float(high)})
        print(f"Test {NAMES[a]} vs {NAMES[b]}: delta={test['delta_macro_f1']:+.4f}, "
              f"p={test['p_raw']:.5f}", flush=True)
    for family in ("primary", "exploratory"):
        selected = [r for r in records if r["family"] == family]
        for record, corrected in zip(selected, holm_adjust([r["p_raw"] for r in selected])):
            record["p_holm"] = corrected
            record["significant_005"] = bool(corrected < 0.05)
    return records


def subgroup_analysis(experiment, metadata, draws, random_seed):
    sarcastic = np.isin(experiment.sarcasm, list(SARCASTIC))
    groups = {"Non-sarcastic": ~sarcastic, "Sarcastic": sarcastic}
    if metadata:
        humorous = np.array([r["humor"] != "not_funny" for r in metadata])
        groups.update({"Not funny": ~humorous, "Humorous": humorous})
    summaries = [{"group": name, **r} for name, mask in groups.items() if np.any(mask)
                 for r in summarize(experiment, mask)]
    bootstraps = {}
    for i, name in enumerate(("Non-sarcastic", "Sarcastic")):
        mask = groups[name]
        if np.any(mask):
            bootstraps[name] = paired_bootstrap(experiment.y[mask],
                {m: p[:, mask] for m, p in experiment.predictions.items()}, draws, random_seed + 500 + i)
    intervals, interactions = [], []
    for model in MODELS[2:]:
        distributions = {}
        for name, bootstrap in bootstraps.items():
            delta = bootstrap[:, MODELS.index(model)] - bootstrap[:, 0]
            distribution = next(r for r in summaries if r["model"] == model and r["group"] == name)
            baseline = next(r for r in summaries if r["model"] == "text" and r["group"] == name)
            low, high = np.quantile(delta, [0.025, 0.975])
            intervals.append({"model": model, "group": name,
                              "delta_macro_f1": distribution["macro_f1_mean"] - baseline["macro_f1_mean"],
                              "ci95_low": float(low), "ci95_high": float(high)})
            distributions[name] = delta
        if len(distributions) == 2:
            interaction = distributions["Sarcastic"] - distributions["Non-sarcastic"]
            low, high = np.quantile(interaction, [0.025, 0.975])
            observed = {r["group"]: r["delta_macro_f1"] for r in intervals if r["model"] == model}
            interactions.append({"model": model,
                                 "delta_sarcastic_minus_nonsarcastic": observed["Sarcastic"] - observed["Non-sarcastic"],
                                 "ci95_low": float(low), "ci95_high": float(high)})
    return groups, summaries, intervals, interactions


def error_analysis(experiment, metadata):
    transitions = []
    text_correct = experiment.predictions["text"] == experiment.y
    for model in MODELS[2:]:
        fusion_correct = experiment.predictions[model] == experiment.y
        gains = (~text_correct & fusion_correct).sum(1)
        losses = (text_correct & ~fusion_correct).sum(1)
        transitions.append({"model": model, "gains_mean": float(gains.mean()),
                            "losses_mean": float(losses.mean()), "gains_seeds": gains.tolist(),
                            "losses_seeds": losses.tolist()})
    # Ensembles below are only a deterministic mechanism for error-case selection.
    ensemble = {m: p.mean(0).argmax(1) for m, p in experiment.probabilities.items()}
    text = ensemble["text"] == experiment.y
    concat = ensemble["both_concat"] == experiment.y
    all_wrong = np.logical_and.reduce([ensemble[m] != experiment.y for m in MODELS])
    categories = [("Fusion hurts", text & ~concat, 6), ("Fusion helps", ~text & concat, 6),
                  ("All five wrong", all_wrong, 8)]
    selected = []
    certainty = np.max(experiment.probabilities["both_concat"].mean(0), axis=1)
    used = set()
    for category, mask, limit in categories + [("Other disagreement", ~text | ~concat, 20)]:
        candidates = sorted(np.flatnonzero(mask), key=lambda i: (-certainty[i], experiment.ids[i]))
        for i in [i for i in candidates if i not in used][:min(limit, 20 - len(selected))]:
            used.add(i)
            selected.append({"id": experiment.ids[i], "category": category,
                             "gold": int(experiment.y[i]), "sarcasm": str(experiment.sarcasm[i]),
                             "humor": metadata[i]["humor"] if metadata else None,
                             **{m: int(ensemble[m][i]) for m in MODELS},
                             "concat_confidence": float(certainty[i])})
    return transitions, selected


def modality_complementarity(experiment):
    """Describe complementary unimodal predictions; no causal feature attribution.

    Categories are assigned separately for each matched seed. Rates below pool
    seed/meme decisions descriptively and are never used as independent test units.
    """
    text = experiment.predictions["text"] == experiment.y
    image = experiment.predictions["image"] == experiment.y
    categories = [("text_only_correct", text & ~image),
                  ("image_only_correct", image & ~text),
                  ("both_correct", text & image), ("both_wrong", ~text & ~image)]
    records = []
    for category, mask in categories:
        counts = mask.sum(1)
        record = {"state": category, "n_mean_per_seed": float(counts.mean()),
                  "n_seeds": counts.tolist(), "decision_count": int(mask.sum()),
                  "share_mean": float(counts.mean() / len(experiment.y))}
        for model in MODELS[2:]:
            numerator = int(((experiment.predictions[model] == experiment.y) & mask).sum())
            record[f"{model}_correct_rate"] = numerator / mask.sum() if mask.sum() else None
        records.append(record)
    return records


def write_csv(path, records):
    if not records:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def plot_all(experiment, summary, comparison, groups, group_summary, transitions, output, complementarity=None):
    """Only the three figures needed by the focused research narrative."""
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    def save(fig, name):
        fig.savefig(figures / f"{name}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    labels = ["Text-only\nBERT", "Image-only\nResNet50", "Both\nConcat", "Both\nCross-Attention", "Both\nProduct"]
    x = np.arange(len(MODELS))
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    means = [r["macro_f1_mean"] for r in summary]
    deviations = [r["macro_f1_sd"] for r in summary]
    ax.bar(x, means, color=COLORS, yerr=deviations, capsize=4)
    for i, value in enumerate(means):
        ax.text(i, value + deviations[i] + 0.01, f"{value:.4f}", ha="center")
    ax.set(title="Image, Text, or Both? Mean Macro-F1 across 3 seeds",
           xticks=x, xticklabels=labels, ylim=(0, 0.45), ylabel="Macro-F1 (higher is better)")
    ax.grid(axis="y", alpha=0.15)
    save(fig, "overall_performance")

    names = ("Non-sarcastic", "Sarcastic")
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    for j, group in enumerate(names):
        rows = [next(r for r in group_summary if r["model"] == model and r["group"] == group) for model in MODELS]
        ax.bar(x + (j - 0.5) * 0.36, [r["macro_f1_mean"] for r in rows], width=0.34,
               yerr=[r["macro_f1_sd"] for r in rows], capsize=3,
               label=f"{group} (n={groups[group].sum()})", color=("#5384bd", "#23a394")[j])
    ax.set(xticks=x, xticklabels=labels, ylabel="Macro-F1", ylim=(0, 0.48),
           title="Does combining image and text help more with sarcasm?")
    ax.legend()
    ax.grid(axis="y", alpha=0.15)
    save(fig, "sarcasm_subgroups")

    if complementarity:
        fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
        descriptions = ["Text correct / Image wrong", "Image correct / Text wrong",
                        "Text and Image correct", "Text and Image wrong"]
        counts = [r["n_mean_per_seed"] for r in complementarity]
        ax.barh(np.arange(4), counts, color=[COLORS[0], COLORS[1], COLORS[2], "#718096"])
        for i, row in enumerate(complementarity):
            ax.text(row["n_mean_per_seed"] + 3, i, f"{row['n_mean_per_seed']:.1f}", va="center")
        ax.set(yticks=np.arange(4), yticklabels=descriptions, xlim=(0, max(counts)*1.2),
               xlabel="Mean number of memes per seed (4 groups sum to 700)",
               title="Image-only and Text-only succeed on different memes")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.15)
        save(fig, "modality_complementarity")


def render_report(*args):
    try:
        from .analysis_report import render_report as build_report
    except ImportError:
        from analysis_report import render_report as build_report
    return build_report(*args)


def run(args):
    output = Path(args.output)
    (output / "tables").mkdir(parents=True, exist_ok=True)
    experiment = load_experiment(args.results, args.config)
    metadata = load_metadata(args.metadata, experiment)
    print(f"Validated {len(experiment.runs)} runs / {len(experiment.y)} aligned memes", flush=True)
    summary = summarize(experiment)
    bootstrap = paired_bootstrap(experiment.y, experiment.predictions, args.bootstrap, args.random_seed)
    print("Overall paired bootstrap complete", flush=True)
    comparison = comparisons(experiment, bootstrap, args.permutations, args.random_seed)
    groups, group_summary, intervals, interactions = subgroup_analysis(experiment, metadata, args.bootstrap, args.random_seed)
    transitions, errors = error_analysis(experiment, metadata)
    complementarity = modality_complementarity(experiment)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    manifest = {"input_commit": commit, "result_sha256": experiment.fingerprints,
                "bootstrap_draws": args.bootstrap, "permutation_draws": args.permutations,
                "random_seed": args.random_seed, "std_ddof": 1,
                "statistic": "mean of seed-wise 3-class Macro-F1",
                "bootstrap": "class-stratified paired meme bootstrap, conditional on saved trained models",
                "permutation": "two-sided, swap all seeds together within each meme",
                "multiple_testing": "Holm: 6 primary comparisons; 4 exploratory comparisons separately",
                "metadata_sha256": hashlib.sha256(Path(args.metadata).read_bytes()).hexdigest() if metadata else None}
    source_path = Path(args.metadata).with_suffix(".source.json") if args.metadata else None
    if source_path and source_path.exists():
        manifest["metadata_source"] = json.loads(source_path.read_text(encoding="utf-8"))
    confusions = [{"model": m, "seed": seed, "actual": CLASSES[i], "predicted": CLASSES[j],
                   "count": int(confusion(experiment.y, experiment.predictions[m])[s, i, j])}
                  for m in MODELS for s, seed in enumerate(experiment.seeds)
                  for i, j in itertools.product(range(3), repeat=2)]
    artifacts = {"input_manifest": manifest, "overall_metrics": summary,
                 "per_seed_metrics": experiment.runs, "paired_comparisons": comparison,
                 "subgroup_metrics": group_summary, "sarcasm_differences": intervals,
                 "sarcasm_interactions": interactions, "error_transitions": transitions,
                 "modality_complementarity": complementarity,
                 "error_candidates": errors, "confusion_matrices": confusions}
    (output / "analysis_summary.json").write_text(json.dumps(artifacts, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, records in artifacts.items():
        if name != "input_manifest":
            write_csv(output / "tables" / f"{name}.csv", records)
    plot_all(experiment, summary, comparison, groups, group_summary, transitions, output, complementarity)
    report = render_report(experiment, summary, comparison, groups, group_summary, intervals, interactions,
                           transitions, errors, metadata, manifest, args.report_date, output, complementarity)
    print(f"Report: {report.resolve()}", flush=True)
    return artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--config", default=None)
    parser.add_argument("--metadata", default="report/inputs/test_metadata.jsonl")
    parser.add_argument("--output", default="report")
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--permutations", type=int, default=20000)
    parser.add_argument("--random-seed", type=int, default=20261007)
    parser.add_argument("--report-date", default=date.today().isoformat())
    args = parser.parse_args()
    if args.bootstrap < 1 or args.permutations < 1:
        parser.error("Resample counts must be positive")
    run(args)


if __name__ == "__main__":
    main()
