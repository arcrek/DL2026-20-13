---
title: "Journal: Role B - Text Baseline (BERT) Execution & Results"
date: 2026-10-06
role: "Role B (Text Baseline Specialist)"
status: completed
model: "bert-base-uncased"
seeds: [0, 1, 2]
---

# Journal: Role B - Text Baseline (BERT) Execution & Results

## 1. Executive Summary
Role B has completed the implementation, training, and evaluation pipeline for the unimodal text baseline (**Text-only Baseline**) utilizing `bert-base-uncased` on the **Memotion 7k** dataset (SemEval-2020 Task 8, Task A: 3-class sentiment).

All experiments were executed in a GPU environment (Tesla T4) via Google Colab. The pipeline strictly enforces a **Differential Learning Rate**, class-weighted cross-entropy loss (`class_weights`), and an early stopping / model selection mechanism based on hold-out Macro-F1. All three test prediction files (700 samples on the `test` split) have been successfully generated and verified against Schema v1 contracts at `results/text_seed{0,1,2}.json`.

---

## 2. Experimental Setup

- **Model Architecture:**
  - Backbone: `bert-base-uncased` (110M parameters).
  - Feature Extraction: Token representation of `[CLS]` via `pooler_output` (768 dimensions).
  - Classification Head: `nn.Sequential(nn.Dropout(0.2), nn.Linear(768, 3))`.
- **Optimization Strategy:**
  - Differential Learning Rate:
    - Backbone (BERT): $lr = 1.5 \times 10^{-5}$
    - Head (Linear): $lr = 5.0 \times 10^{-4}$
    - Weight decay: $0.01$
  - Learning Rate Scheduler: Linear Warmup (10% of total training steps) followed by Linear Decay.
  - Batch size: 32 | Max token length: 128 (with dynamic padding and truncation).
- **Leakage Prevention & Class Imbalance Handling:**
  - Training data was split into `fit` (5,034 samples) and `holdout` (559 samples) using Union-Find template grouping to prevent meme template leakage.
  - Class weights: `class_weights = [3.656, 1.060, 0.561]` computed on the `fit` split via `features/train_holdout.json` (corresponding to `[3.616, 1.051, 0.564]` on the full train split) for the three classes (0: Negative ~9%, 1: Neutral ~31%, 2: Positive ~59%).
  - Model Selection: Best checkpoint saved at the epoch achieving the highest Macro-F1 on the `holdout` split.

---

## 3. Quantitative Results

Evaluation metrics on the test split (700 samples) across three independent random seeds:

| Run / Seed | Best Epoch (Holdout) | Holdout Macro-F1 | Test Accuracy | Test Macro-F1 | File Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Seed 0** | Epoch 2 | 0.3337 | 0.3914 | **0.3283** | `results/text_seed0.json` |
| **Seed 1** | Epoch 5 | 0.2942 | 0.3543 | **0.2988** | `results/text_seed1.json` |
| **Seed 2** | Epoch 1 | 0.3426 | 0.5500 | **0.3677** | `results/text_seed2.json` |
| **Mean ± Std** | — | **0.3235 ± 0.0258** | **0.4319 ± 0.1039** | **0.3316 ± 0.0346** | Validated Schema v1 |

*(Note: Sample standard deviation calculated with $ddof=1$. Population standard deviation with $ddof=0$ yields Accuracy $0.4319 \pm 0.0849$ and Macro-F1 $0.3316 \pm 0.0282$)*.

---

## 4. Key Findings & Error Patterns

1. **Convergence Characteristics of BERT on Meme Text:**
   - On **Seed 2**, the model attained peak Macro-F1 (0.3677) and Accuracy (0.5500) as early as **Epoch 1**. This demonstrates the strong pre-trained semantic representations of BERT: fine-tuning the upper layers allows rapid adaptation without prolonged training epochs.
2. **Class Imbalance Challenges:**
   - Despite utilizing `class_weights`, the minority Negative class (~9%) remains the primary bottleneck, constraining average Macro-F1 to ~0.33. Text-only models frequently confuse Negative and Neutral instances when sarcastic memes deploy superficially positive or neutral phrasing.
3. **Value as a Modality Baseline:**
   - These findings establish a rigorous baseline for **Role E (Analysis)** to perform hypothesis testing (**Paired Bootstrap / Permutation Test**): evaluating whether Multimodal Fusion (Role D) achieves statistically significant improvements ($p < 0.05$) over the text-only Macro-F1 baseline of 0.3316.

---

## 5. Downstream Handoff

- [x] **Role A (Tech Lead):** Standard JSON files provided at `results/text_seed{0,1,2}.json`, passing 100% of test suites in `tests/test_results.py`.
- [x] **Role E (Analysis):** Output probability distributions (`probs`), true labels (`y_true`), predicted labels (`y_pred`), and sarcasm tags (`sarcasm`) ready for:
  - Sub-group Sarcasm Analysis.
  - Confusion Matrix generation.
  - Statistical comparison against Image Baseline (Role C) and Fusion Baseline (Role D).
- [x] **Role F (Report):** Quantitative metrics table provided in Section 3 and model architecture details prepared for Section 4 & 5 of the technical report.
- [x] **Source Code (`src/`):** Delivered clean implementation in `src/models/text.py` with CLI entrypoint at `src/text.py`.
