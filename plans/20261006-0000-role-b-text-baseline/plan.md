---
title: "Role B: Text Baseline Implementation & Training Plan"
status: completed
priority: P1
effort: "12h"
blockedBy: []
blocks: ["20261001-0830-meme-understanding-modality-study"]
---

# Role B: Text Baseline Implementation & Training Plan

## Executive Summary
Detailed implementation plan for **Role B (Text Baseline Specialist)** in the research project *"Multimodal Meme Understanding: Image, Text, or Both?"* on the **Memotion 7k** dataset (SemEval-2020 Task 8, Task A).

Role B is responsible for constructing, training, and evaluating the unimodal text baseline (**Text-only Baseline**) using `bert-base-uncased` with **Differential Learning Rate**, training across 3 fixed random seeds (0, 1, 2), handling class imbalance via `class_weights`, and exporting results adhering strictly to the contract schema at `results/text_seed{0,1,2}.json` to directly serve hypothesis testing for Role E and reporting for Role F.

## Context & Constraints
- **Roadmap Reference:** [docs/project-roadmap.md](../../docs/project-roadmap.md) §Phase 2 (Track 1).
- **Team Split:** [plans/reports/advise-260510-team-split.md](../reports/advise-260510-team-split.md).
- **Configuration:** [configs/base.yaml](../../configs/base.yaml) (`configs.text: B`, `seeds: [0, 1, 2]`, `class_weighted_loss: true`).
- **Results Contract:** [src/results.py](../../src/results.py) (`save_results()`, schema version 1).
- **Timeline:** Hours 6 - 18 (complete training), Hours 18 - 34 (support analysis with Role E), Hours 34 - 48 (support reporting with Role F and integration with Role A).

## Phase Overview
| Phase | Title | Effort | Priority | Deliverables |
|---|---|---|---|---|
| [Phase 1](phase-01-scaffold-and-data.md) | Module Scaffold & Data Pipeline | 2h | P1 | `src/text.py` & `src/models/text.py` (Dataset, DataLoader, Collate) |
| [Phase 2](phase-02-model-and-optimizer.md) | Model Architecture & Differential LR | 2h | P1 | `BertMemeClassifier`, AdamW Differential LR, Loss weighting |
| [Phase 3](phase-03-training-and-results.md) | Training Loop & Contract Export (3 Seeds) | 5h | P1 | `results/text_seed{0,1,2}.json`, Checkpoint selection |
| [Phase 4](phase-04-downstream-and-report.md) | Downstream Handoff & Report Contribution | 3h | P2 | Sarcasm Analysis, Paired Bootstrap test, Sections 4 & 5 report |

## Success Metrics
- 3 valid JSON files at `results/text_seed0.json`, `results/text_seed1.json`, `results/text_seed2.json`.
- `pytest tests/test_results.py` passes without exceptions.
- Summary table with `mean ± std` Macro-F1 and Accuracy of Text-only baseline prepared for Role E and F.
- Strict data hygiene maintained (no use of `validation`/`test` for model selection; only `holdout` is used).
