---
title: "Role D: Multimodal Fusion Implementation & Training Plan"
status: in_progress
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
- **Hardware:** Local NVIDIA GeForce RTX 3060 Laptop GPU (6GB VRAM) with mixed precision (bfloat16 / fp16) or Colab/Kaggle GPU.

## Phase Overview
| Phase | Title | Effort | Priority | Deliverables |
|---|---|---|---|---|
| Phase 1 | Scaffold & Architecture Design | 3h | P1 | `src/models/fusion.py`, `src/fusion.py`, `tests/test_fusion.py` |
| Phase 2 | Environment & Data Pipeline | 2h | P1 | `.venv`, `scripts/download_data.sh`, `features/train_holdout.json` |
| Phase 3 | Model Training & Result Export | 8h | P1 | 9 JSON files in `results/both_*.json` (3 configs x 3 seeds) |
| Phase 4 | Notebook Integration & Downstream Handoff | 3h | P2 | `notebooks/meme_understanding.ipynb`, summary table, handoff to Role E & F |

## Success Metrics
- Unit tests pass: `pytest tests/test_results.py tests/test_fusion.py`.
- 9 valid JSON files in `results/`:
  - `both_concat_seed{0,1,2}.json`
  - `both_cross_attn_seed{0,1,2}.json`
  - `both_product_seed{0,1,2}.json`
- Strict compliance with `schema_version = 1`.
- Clean reproducibility on Google Colab / Kaggle.
