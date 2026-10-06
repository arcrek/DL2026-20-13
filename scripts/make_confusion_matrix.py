#!/usr/bin/env python3
"""Generate confusion matrices and error analysis from evaluation results.

Reads result JSON files from results/ (e.g. text_seed0.json, both_cross_attn_seed1.json)
and produces:
1. Formatted terminal/markdown confusion matrices and classification reports.
2. High-resolution publication-quality plots (all models comparative grid & individual figures).
3. Sarcasm-stratified confusion matrices to analyze rhetorical performance gaps.

Usage:
    python scripts/make_confusion_matrix.py
    python scripts/make_confusion_matrix.py --output-dir report/figures --results-dir results
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

CLASS_NAMES = ["Negative", "Neutral", "Positive"]
DEFAULT_CONFIGS = ["text", "image", "both_concat", "both_product", "both_cross_attn"]
CONFIG_LABELS = {
    "text": "Text-only (BERT)",
    "image": "Image-only (ResNet-50)",
    "both_concat": "Late Concat Fusion",
    "both_product": "Product Fusion",
    "both_cross_attn": "Cross-Attention Fusion",
}
SEEDS = [0, 1, 2]


def load_model_results(
    config_name: str,
    seeds: List[int] = SEEDS,
    results_dir: str = "results",
) -> Dict[str, list]:
    """Load results for a specific model config across multiple seeds."""
    combined = {
        "ids": [],
        "y_true": [],
        "y_pred": [],
        "probs": [],
        "sarcasm": [],
        "seed_accuracies": [],
        "seed_f1s": [],
        "seed_cms": [],
    }

    for seed in seeds:
        path = os.path.join(results_dir, f"{config_name}_seed{seed}.json")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        combined["ids"].extend(data["ids"])
        combined["y_true"].extend(data["y_true"])
        combined["y_pred"].extend(data["y_pred"])
        combined["probs"].extend(data["probs"])
        combined["sarcasm"].extend(data["sarcasm"])
        combined["seed_accuracies"].append(data["accuracy"])
        combined["seed_f1s"].append(data["macro_f1"])

        cm = confusion_matrix(data["y_true"], data["y_pred"], labels=[0, 1, 2])
        combined["seed_cms"].append(cm)

    return combined


def compute_cm_stats(y_true: List[int], y_pred: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute raw confusion matrix and row-normalized (recall) matrix."""
    raw_cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    row_sums = raw_cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm_cm = np.where(row_sums > 0, raw_cm / row_sums, 0.0)
    return raw_cm, norm_cm


def print_ascii_cm(raw_cm: np.ndarray, norm_cm: np.ndarray, title: str):
    """Print a clean textual confusion matrix with counts and percentages."""
    print(f"\n{'=' * 65}")
    print(f" {title}")
    print(f"{'=' * 65}")
    header = f"{'True \\ Pred':<14} | " + " | ".join(f"{c:>12}" for c in CLASS_NAMES) + " | Total"
    print(header)
    print("-" * len(header))
    for i, true_name in enumerate(CLASS_NAMES):
        row_str = f"{true_name:<14} | "
        for j in range(3):
            count = raw_cm[i, j]
            pct = norm_cm[i, j] * 100
            cell = f"{count:>4} ({pct:>4.1f}%)"
            row_str += f"{cell:>12} | "
        row_str += f"{raw_cm[i].sum():>5}"
        print(row_str)
    print("-" * len(header))
    pred_totals = f"{'Total Pred':<14} | "
    for j in range(3):
        pred_totals += f"{raw_cm[:, j].sum():>12} | "
    pred_totals += f"{raw_cm.sum():>5}"
    print(pred_totals)


def plot_single_cm(
    raw_cm: np.ndarray,
    norm_cm: np.ndarray,
    title: str,
    output_path: str,
    subtitle: Optional[str] = None,
):
    """Plot an individual high-quality confusion matrix heatmap."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(6, 5), dpi=300)

    # Format annotations: count on line 1, percentage on line 2
    annot = np.empty_like(raw_cm, dtype=object)
    for i in range(3):
        for j in range(3):
            annot[i, j] = f"{raw_cm[i, j]}\n({norm_cm[i, j] * 100:.1f}%)"

    sns.heatmap(
        norm_cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        vmin=0.0,
        vmax=1.0,
        linewidths=1.0,
        linecolor="white",
        cbar_kws={"label": "Normalized Recall"},
    )

    full_title = title if not subtitle else f"{title}\n{subtitle}"
    plt.title(full_title, fontsize=12, pad=12, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=11, labelpad=8)
    plt.ylabel("Ground Truth Class", fontsize=11, labelpad=8)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_comparative_grid(
    configs: List[str],
    results_dir: str,
    output_path: str,
    seeds: List[int] = SEEDS,
):
    """Plot all models in a comparative grid showing raw counts and recalls."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    n_models = len(configs)
    fig, axes = plt.subplots(1, n_models, figsize=(4.2 * n_models, 4.5), dpi=300)
    if n_models == 1:
        axes = [axes]

    for idx, cfg in enumerate(configs):
        res = load_model_results(cfg, seeds=seeds, results_dir=results_dir)
        if not res["y_true"]:
            continue
        raw_cm, norm_cm = compute_cm_stats(res["y_true"], res["y_pred"])

        annot = np.empty_like(raw_cm, dtype=object)
        for i in range(3):
            for j in range(3):
                annot[i, j] = f"{raw_cm[i, j]}\n({norm_cm[i, j] * 100:.1f}%)"

        ax = axes[idx]
        is_last = idx == n_models - 1
        sns.heatmap(
            norm_cm,
            annot=annot,
            fmt="",
            cmap="Blues",
            cbar=is_last,
            ax=ax,
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES if idx == 0 else False,
            vmin=0.0,
            vmax=1.0,
            linewidths=0.8,
            linecolor="white",
            cbar_kws={"label": "Recall (Row %)"} if is_last else None,
        )

        mean_acc = np.mean(res["seed_accuracies"]) if res["seed_accuracies"] else 0
        mean_f1 = np.mean(res["seed_f1s"]) if res["seed_f1s"] else 0
        ax.set_title(
            f"{CONFIG_LABELS.get(cfg, cfg)}\nAcc: {mean_acc:.3f} | Macro-F1: {mean_f1:.3f}",
            fontsize=11,
            pad=10,
            fontweight="semibold",
        )
        ax.set_xlabel("Predicted", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Ground Truth", fontsize=10)

    fig.suptitle(
        f"Comparative Confusion Matrices on Memotion 7k Test Set (Pooled over Seeds {seeds})",
        fontsize=14,
        fontweight="bold",
        y=1.05,
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_sarcasm_stratified_cm(
    configs: List[str],
    results_dir: str,
    output_path: str,
    seeds: List[int] = SEEDS,
):
    """Plot confusion matrices stratified by Sarcastic vs Non-Sarcastic memes."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(2, len(configs), figsize=(4.2 * len(configs), 8.5), dpi=300)

    for col_idx, cfg in enumerate(configs):
        res = load_model_results(cfg, seeds=seeds, results_dir=results_dir)
        if not res["y_true"]:
            continue

        y_true = np.array(res["y_true"])
        y_pred = np.array(res["y_pred"])
        sarcasm = np.array(res["sarcasm"])

        is_sarcastic = sarcasm != "not_sarcastic"

        for row_idx, (subset_mask, subset_name) in enumerate(
            [(~is_sarcastic, "Non-Sarcastic"), (is_sarcastic, "Sarcastic (Irony/Metaphor)")]
        ):
            sub_true = y_true[subset_mask]
            sub_pred = y_pred[subset_mask]
            raw_cm, norm_cm = compute_cm_stats(sub_true.tolist(), sub_pred.tolist())

            annot = np.empty_like(raw_cm, dtype=object)
            for i in range(3):
                for j in range(3):
                    annot[i, j] = f"{raw_cm[i, j]}\n({norm_cm[i, j] * 100:.1f}%)"

            ax = axes[row_idx, col_idx]
            is_last_col = col_idx == len(configs) - 1
            sns.heatmap(
                norm_cm,
                annot=annot,
                fmt="",
                cmap="YlGnBu",
                cbar=is_last_col,
                ax=ax,
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES if col_idx == 0 else False,
                vmin=0.0,
                vmax=1.0,
                linewidths=0.8,
                linecolor="white",
                cbar_kws={"label": "Recall"} if is_last_col else None,
            )

            sub_acc = (sub_true == sub_pred).mean() if len(sub_true) else 0.0
            ax.set_title(
                f"{CONFIG_LABELS.get(cfg, cfg)}\n[{subset_name}] (N={len(sub_true)}, Acc: {sub_acc:.3f})",
                fontsize=9.5,
                pad=8,
            )
            ax.set_xlabel("Predicted", fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(f"{subset_name}\nGround Truth", fontsize=10)

    fig.suptitle(
        "Error Analysis: Confusion Matrices Stratified by Sarcasm / Irony",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def generate_markdown_summary(configs: List[str], results_dir: str, seeds: List[int] = SEEDS) -> str:
    """Generate Markdown tables of confusion matrices and performance metrics."""
    lines = []
    lines.append("# Memotion 7k Confusion Matrix & Error Analysis Summary\n")
    lines.append(f"Evaluated on test split across seeds {seeds}.\n")

    for cfg in configs:
        res = load_model_results(cfg, seeds=seeds, results_dir=results_dir)
        if not res["y_true"]:
            continue
        raw_cm, norm_cm = compute_cm_stats(res["y_true"], res["y_pred"])
        mean_acc = np.mean(res["seed_accuracies"]) if res["seed_accuracies"] else 0.0
        mean_f1 = np.mean(res["seed_f1s"]) if res["seed_f1s"] else 0.0

        lines.append(f"### {CONFIG_LABELS.get(cfg, cfg)}")
        lines.append(f"- **Mean Accuracy:** `{mean_acc:.4f}`")
        lines.append(f"- **Mean Macro-F1:** `{mean_f1:.4f}`\n")

        lines.append("| True \\ Predicted | Negative (0) | Neutral (1) | Positive (2) | Total Support |")
        lines.append("|---|---|---|---|---|")
        for i, cname in enumerate(CLASS_NAMES):
            row = [f"**{cname}**"]
            for j in range(3):
                row.append(f"{raw_cm[i, j]} ({norm_cm[i, j] * 100:.1f}%)")
            row.append(f"{raw_cm[i].sum()}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("\n")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate confusion matrices from evaluation results.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory containing result JSON files.")
    parser.add_argument("--output-dir", type=str, default="report/figures", help="Directory to save figures.")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS, help="Random seeds to evaluate.")
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        default=DEFAULT_CONFIGS,
        help="Model configurations to include.",
    )
    parser.add_argument(
        "--save-individual",
        action="store_true",
        default=True,
        help="Whether to save individual CM plots per model.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading results from '{args.results_dir}' for configs: {args.configs}, seeds: {args.seeds}...")

    # 1. Print Text Confusion Matrices and Reports
    for cfg in args.configs:
        res = load_model_results(cfg, seeds=args.seeds, results_dir=args.results_dir)
        if not res["y_true"]:
            print(f"Warning: No results found for config {cfg}")
            continue
        raw_cm, norm_cm = compute_cm_stats(res["y_true"], res["y_pred"])
        mean_acc = np.mean(res["seed_accuracies"]) if res["seed_accuracies"] else 0.0
        mean_f1 = np.mean(res["seed_f1s"]) if res["seed_f1s"] else 0.0
        label = CONFIG_LABELS.get(cfg, cfg)

        print_ascii_cm(
            raw_cm,
            norm_cm,
            f"{label} (Mean Acc: {mean_acc:.4f}, Mean F1: {mean_f1:.4f})",
        )
        print("\nClassification Report (Pooled over seeds):")
        print(classification_report(res["y_true"], res["y_pred"], target_names=CLASS_NAMES, digits=4))

        if args.save_individual:
            individual_path = os.path.join(args.output_dir, f"cm_{cfg}.png")
            plot_single_cm(
                raw_cm,
                norm_cm,
                label,
                individual_path,
                subtitle=f"Accuracy: {mean_acc:.3f} | Macro-F1: {mean_f1:.3f} (Seeds {args.seeds})",
            )
            print(f"Saved: {individual_path}")

    # 2. Comparative Grid Plot
    grid_path = os.path.join(args.output_dir, "confusion_matrices_all.png")
    plot_comparative_grid(args.configs, args.results_dir, grid_path, seeds=args.seeds)
    print(f"\nSaved comparative grid figure to: {grid_path}")

    # PDF format for LaTeX inclusion
    grid_pdf_path = os.path.join(args.output_dir, "confusion_matrices_all.pdf")
    plot_comparative_grid(args.configs, args.results_dir, grid_pdf_path, seeds=args.seeds)
    print(f"Saved PDF vector figure for LaTeX report to: {grid_pdf_path}")

    # 3. Sarcasm Stratified Confusion Matrix Plot
    sarcasm_path = os.path.join(args.output_dir, "confusion_matrices_sarcasm.png")
    plot_sarcasm_stratified_cm(args.configs, args.results_dir, sarcasm_path, seeds=args.seeds)
    print(f"Saved sarcasm-stratified confusion matrices to: {sarcasm_path}")

    # 4. Generate Markdown Summary
    md_summary = generate_markdown_summary(args.configs, args.results_dir, seeds=args.seeds)
    summary_path = os.path.join(args.output_dir, "confusion_matrix_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(md_summary)
    print(f"Saved Markdown summary to: {summary_path}")


if __name__ == "__main__":
    main()
