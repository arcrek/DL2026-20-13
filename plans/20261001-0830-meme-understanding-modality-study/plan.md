---
title: "Multimodal Meme Understanding: Image, Text, or Both?"
---

# Multimodal Meme Understanding: Image, Text, or Both?

Dataset: Memotion 7k (SemEval-2020 Task 8: Memotion Analysis, Sharma et al., 2020).
Compute: Colab (A100/H100 or T4).

## Research Questions
1. How do visual and textual information individually contribute to understanding the meaning/sentiment of memes?
2. Does fusing both modalities outperform unimodal models (image-only vs text-only)?
3. Under what conditions (e.g. sentiment ambiguity, sarcasm/humor interaction) does multimodal fusion yield the largest gains?

## Phases
| # | Phase | Hours | Status |
|---|-------|-------|--------|
| 1 | [Data and features](phase-01-data-and-features.md) | 0-6 | pending |
| 2 | [Baselines and main models](phase-02-models.md) | 6-18 | pending |
| 3 | [Analysis, ablations, errors](phase-03-analysis.md) | 18-34 | pending |
| 4 | [Repo, report, slides, submit](phase-04-deliverables.md) | 34-48 | pending |

## Key Decisions
- Frozen CLIP ViT-L/14 features cached once; light heads, 5 seeds.
- Matched probes (CLIP image vs CLIP text) so gaps reflect modality, not encoder quality.
- Hyperparameters tuned on a group-aware train hold-out (no shared caption/image), never on validation/test.
- Evaluation metrics: Macro F1-score & Accuracy (official SemEval-2020 Task A metrics).
- Setup 1 = Unimodal Baselines vs Multimodal Fusion.
- Setup 2 = Modality Contribution (Image vs Text vs Both across categories: Sarcasm, Humor, Motivation).
- Setup 3 = Fusion Architectures Ablation (Concat vs Product vs Gated Fusion) & Error Analysis.

## Success Metrics
- Image, text, and fusion results over 5 seeds (mean ± std) on identical splits.
- Significance test (paired bootstrap or permutation test) for each fusion-vs-unimodal claim.
- `notebooks/meme_understanding.ipynb` reproduces the complete pipeline from a clean Colab.
- Setup1/2/3 each have a table or figure plus written interpretation.
- At least 20 categorized qualitative error cases.
- Report 10-15 pages following the exam template; submitted before the deadline.
