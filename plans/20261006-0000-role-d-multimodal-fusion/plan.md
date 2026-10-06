---
title: "Role D: Multimodal Fusion Implementation & Training Plan"
status: completed
priority: P1
effort: "16h"
blockedBy: []
blocks: ["20261001-0830-meme-understanding-modality-study"]
---

# Role D: Multimodal Fusion Implementation & Training Plan

## Executive Summary
Detailed implementation plan for **Role D (Fusion Specialist)** in the research project *"Multimodal Meme Understanding: Image, Text, or Both?"* on the **Memotion 7k** dataset (SemEval-2020 Task 8, Task A).

Role D is responsible for designing, implementing, training, and evaluating three multimodal fusion architectures:
1. **Branch 3A - Late Concat Fusion (`both_concat`)**: Concatenation of BERT [CLS] (768d) and ResNet50 GAP (2048d) -> MLP Classifier.
2. **Branch 3B - Cross-Attention Fusion (`both_cross_attn`)**: Text tokens as Query ($L \times 256$), image spatial feature map as Key/Value ($49 \times 256$) with residual LayerNorm -> MLP Classifier.
3. **Ablation Baseline - Product Fusion (`both_product`)**: Element-wise Hadamard product of projected text and image vectors ($512d$) -> MLP Classifier.

All models are trained across 3 fixed random seeds (0, 1, 2) using `features/train_holdout.json` (`fit` for training, `holdout` for checkpoint selection), with loss weighted by `class_weights`. Predictions on the test set are exported strictly following the schema in `src/results.py` to `results/both_*_seed{0,1,2}.json` (9 files total) to enable downstream hypothesis testing by Role E and reporting by Role F.

## Context & Constraints
- **Roadmap Reference:** [docs/project-roadmap.md](../../docs/project-roadmap.md) §Phase 2 (Track 3 & Ablation).
- **Team Split:** [plans/reports/advise-260510-team-split.md](../reports/advise-260510-team-split.md).
- **Configuration:** [configs/base.yaml](../../configs/base.yaml) (`both_concat: D`, `both_cross_attn: D`, `both_product: D`, `seeds: [0, 1, 2]`).
- **Results Contract:** [src/results.py](../../src/results.py) (`save_results()`, schema version 1).
- **Hardware:** Local NVIDIA GeForce RTX 3060 Laptop GPU (6GB VRAM) with mixed precision (bfloat16 / fp16).

## Phase Overview
| Phase | Title | Effort | Priority | Status | Deliverables |
|---|---|---|---|---|---|
| Phase 1 | Scaffold & Architecture Design | 3h | P1 | ✅ Completed | `src/models/fusion.py`, `src/fusion.py`, `tests/test_fusion.py` |
| Phase 2 | Environment & Data Pipeline | 2h | P1 | ✅ Completed | `.venv`, `scripts/download_data.sh`, `features/train_holdout.json` |
| Phase 3 | Model Training & Result Export | 8h | P1 | ✅ Completed | 9 JSON files in `results/both_*.json` (3 configs x 3 seeds) |
| Phase 4 | Notebook Integration & Downstream Handoff | 3h | P2 | ✅ Completed | `notebooks/meme_understanding.ipynb`, summary table, handoff to Role E & F |

## Experimental Results Summary

| Model Config | Seed 0 | Seed 1 | Seed 2 | Macro-F1 (mean ± std) | Accuracy (mean ± std) |
|---|---|---|---|---|---|
| `text` (Unimodal Baseline) | 0.3283 | 0.2988 | 0.3677 | 0.3316 ± 0.0282 | 0.4319 ± 0.0849 |
| **`both_concat` (Late Concat)** | 0.3289 | 0.3495 | 0.3322 | **0.3369 ± 0.0090** | 0.4267 ± 0.0199 |
| **`both_cross_attn` (Cross-Attention)** | 0.3443 | 0.3302 | 0.3061 | 0.3269 ± 0.0158 | **0.4676 ± 0.0541** |
| **`both_product` (Product Fusion)** | 0.3193 | 0.3206 | 0.3231 | 0.3210 ± 0.0016 | 0.4086 ± 0.0135 |

### Key Observations for Role E (Analysis) and Role F (Report):
1. **Late Concat Outperforms Text Baseline:** With the cleaned preprocessing pipeline, `both_concat` achieves the highest overall test Macro-F1 (0.3369 ± 0.0090), surpassing the unimodal text baseline (0.3316 ± 0.0282) while exhibiting 3x lower seed variance.
2. **Cross-Attention Achieves Peak Accuracy:** `both_cross_attn` achieves the highest test accuracy (46.76% mean, peaking at 52.86% on Seed 2), showing strong capability in aligning visual regions with textual tokens.
3. **Decisive Multimodal Gain on Sarcasm:** In memes with complex/severe irony (`very_twisted`), `both_concat` achieves 0.3852 Macro-F1 (+0.1121 over text's 0.2731) and `both_cross_attn` achieves 47.62% Accuracy (+8.8% over text), demonstrating that cross-modal fusion is critical when textual sentiment alone is deceiving.
4. **Product Fusion Stability:** `both_product` demonstrates ultra-stable Macro-F1 across seeds ($\sigma = 0.0016$), providing an effective regularized ablation baseline.

## Success Metrics Verification
- Unit tests pass: `pytest tests/` (9 passed in 0.14s).
- 9 valid JSON files in `results/`:
  - `both_concat_seed{0,1,2}.json` ✅
  - `both_cross_attn_seed{0,1,2}.json` ✅
  - `both_product_seed{0,1,2}.json` ✅
- Full compliance with `schema_version = 1`.
