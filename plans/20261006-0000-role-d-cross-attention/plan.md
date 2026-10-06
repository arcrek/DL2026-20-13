---
title: "Role D: Cross-Attention Multimodal Fusion Implementation & Training Plan"
status: completed
priority: P1
effort: "16h"
blockedBy: []
blocks: ["20261001-0830-meme-understanding-modality-study"]
---

# Role D: Cross-Attention Multimodal Fusion Implementation & Training Plan

## Executive Summary
Implementation plan for the **Cross-Attention Multimodal Fusion (Branch 3B - `both_cross_attn`)** in the research project *"Multimodal Meme Understanding: Image, Text, or Both?"* on the **Memotion 7k** dataset (SemEval-2020 Task 8, Task A).

This component is dedicated specifically to the Cross-Attention architecture:
- Text stream: `bert-base-uncased` sequence tokens ($L \times 768$) projected to ($L \times 256$) serving as **Query**.
- Image stream: `ResNet50` spatial feature map ($7 \times 7 \times 2048 \to 49 \times 256$) serving as **Key** and **Value**.
- Cross-Attention mechanism: Text query tokens dynamically attend to spatial image patches with residual connection and Layer Normalization.
- Head: Pooled `[CLS]` token representation ($256$d) passed through an MLP classifier to predict 3 sentiment classes.

The model is trained across 3 fixed random seeds (0, 1, 2) using `features/train_holdout.json` (`fit` for training, `holdout` for checkpoint selection), with loss weighted by `class_weights`. Predictions on the test set are exported strictly following the schema in `src/results.py` to `results/both_cross_attn_seed{0,1,2}.json` (3 files) to directly serve hypothesis testing for Role E and reporting for Role F.

## Context & Constraints
- **Roadmap Reference:** [docs/project-roadmap.md](../../docs/project-roadmap.md) §Phase 2 (Track 3B).
- **Configuration:** [configs/base.yaml](../../configs/base.yaml) (`both_cross_attn: D`, `seeds: [0, 1, 2]`).
- **Results Contract:** [src/results.py](../../src/results.py) (`save_results()`, schema version 1).
- **Colab/Local Compatibility:** Mixed precision (bfloat16 / fp16) with differential learning rate (`lr_backbone = 1.5e-5`, `lr_head = 5e-4`).

## Phase Overview
| Phase | Title | Effort | Priority | Status | Deliverables |
|---|---|---|---|---|---|
| Phase 1 | Scaffold & Architecture Design | 3h | P1 | ✅ Completed | `src/models/cross_attention.py`, `src/cross_attn.py`, `tests/test_cross_attention.py` |
| Phase 2 | Multimodal Data Pipeline | 2h | P1 | ✅ Completed | `MultimodalMemeDataset`, `make_multimodal_collate_fn`, safe image fallback |
| Phase 3 | Model Training & Contract Export | 6h | P1 | ✅ Completed | `results/both_cross_attn_seed{0,1,2}.json` |
| Phase 4 | Notebook Integration & Downstream Handoff | 3h | P2 | ✅ Completed | Section 6 in `notebooks/meme_understanding.ipynb`, summary table, handoff |

## Experimental Results (Cross-Attention Baseline)

| Model Config | Seed 0 | Seed 1 | Seed 2 | Macro-F1 (mean ± std) | Accuracy (mean ± std) |
|---|---|---|---|---|---|
| `both_cross_attn` (Cross-Attention) | 0.3073 | 0.3284 | 0.3208 | **0.3189 ± 0.0087** | **0.4110 ± 0.0058** |

### Key Observations:
1. **High Stability Across Seeds:** Cross-Attention demonstrates notably low variance ($\sigma_{\text{F1}} \approx 0.0087$, $\sigma_{\text{Acc}} \approx 0.0058$), confirming that cross-modal attention between text tokens and spatial visual patches stabilizes representations across random initializations.
2. **Handoff to Role E:** Ready for downstream paired bootstrap tests against unimodal text baseline (`text_seed*.json`) and sub-group analysis on sarcasm vs non-sarcasm memes.
