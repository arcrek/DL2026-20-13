---
title: "Journal: Role D - Multimodal Fusion Execution & Results"
date: 2026-10-06
role: "Role D (Fusion Specialist)"
status: completed
models: ["both_concat", "both_cross_attn", "both_product"]
seeds: [0, 1, 2]
---

# Journal: Role D - Multimodal Fusion Execution & Results

## 1. Executive Summary
Role D has successfully implemented, trained, and verified the entire multimodal fusion pipeline for **Memotion 7k** (SemEval-2020 Task 8, Task A: 3-class sentiment analysis).

All three multimodal fusion configurations across all three required seeds ($3 \times 3 = 9$ runs) were executed on a local GPU (NVIDIA GeForce RTX 3060 Laptop GPU, 6GB VRAM) with mixed precision. All 9 test prediction files (700 samples on the `test` split) have been generated and validated against Schema v1 contracts in `results/`.

---

## 2. Experimental Setup

- **Architectures:**
  1. **Branch 3A (Late Concat Fusion):**
     - Text: BERT `[CLS]` token (768d).
     - Image: ResNet50 Global Average Pooling (2048d).
     - Fusion: Vector concatenation $[v_{\text{text}}, v_{\text{img}}] \in \mathbb{R}^{2816} \to \text{Linear}(2816, 512) \to \text{ReLU} \to \text{Dropout}(0.2) \to \text{Linear}(512, 3)$.
  2. **Branch 3B (Cross-Attention Fusion):**
     - Query: Projected text sequence tokens ($L \times 256$).
     - Key/Value: Projected spatial conv5 feature map ($49 \times 256$).
     - Fusion: Multi-Head Cross-Attention (4 heads) $\to$ Residual LayerNorm $\to$ [CLS] pooling $\to$ MLP $(256 \to 128 \to 3)$.
  3. **Ablation Baseline (Product Fusion):**
     - Projections: $W_t v_{\text{text}} \in \mathbb{R}^{512}$ and $W_i v_{\text{img}} \in \mathbb{R}^{512}$.
     - Fusion: Element-wise product $v_{\text{fused}} = \text{LayerNorm}(W_t v_{\text{text}} \odot W_i v_{\text{img}}) \to \text{Dropout} \to \text{Linear}(512, 256) \to \text{ReLU} \to \text{Linear}(256, 3)$.

- **Optimization:**
  - Optimizer: AdamW (Backbone LR $1.5 \times 10^{-5}$, Head LR $5 \times 10^{-4}$, weight decay $0.01$).
  - Warmup: 10% linear warmup followed by linear decay.
  - Loss function: Class-weighted CrossEntropyLoss (`class_weights` from `features/train_holdout.json`).
  - Model selection: Peak Macro-F1 on `holdout` split (559 samples).

---

## 3. Comprehensive Experimental Results

| Model Config | Seed 0 | Seed 1 | Seed 2 | Macro-F1 (mean ± std) | Accuracy (mean ± std) |
|---|---|---|---|---|---|
| `text` (Unimodal Baseline) | 0.3283 | 0.2988 | 0.3677 | 0.3316 ± 0.0282 | 0.4319 ± 0.0849 |
| **`both_concat` (Late Concat)** | 0.3112 | 0.3253 | 0.3212 | **0.3192 ± 0.0059** | 0.4262 ± 0.0141 |
| **`both_cross_attn` (Cross-Attention)** | 0.3073 | 0.3284 | 0.3208 | **0.3189 ± 0.0087** | 0.4110 ± 0.0058 |
| **`both_product` (Product Fusion)** | 0.3233 | 0.3507 | 0.3161 | **0.3301 ± 0.0149** | **0.4433 ± 0.0315** |

---

## 4. Key Takeaways & Handoff to Roles E & F
- **Role E (Statistical Analysis):** All 9 JSON files are available in `results/` containing `ids`, `y_true`, `y_pred`, `probs`, and `sarcasm` fields. They are ready for paired bootstrap tests against `text_seed*.json` and sub-group sarcasm evaluations.
- **Role F (Report Writing):** Section 4 (Architecture description) and Section 5 (Quantitative results) can directly reference the table and stability findings above.
