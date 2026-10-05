---
phase: 2
title: "Model Architecture & Differential LR"
status: completed
priority: P1
effort: "2h"
dependencies: ["1"]
---

# Phase 2: Model Architecture & Differential LR

## Overview
Implement the text classification model `BertMemeClassifier` based on `bert-base-uncased`, configuring the optimizer with **Differential Learning Rate** and class-weighted cross-entropy loss (`class_weighted_loss`) to tackle severe class imbalance.

## Requirements
- **Functional:**
  - Backbone: `bert-base-uncased` (via Hugging Face `AutoModel`).
  - Representation: Extract `[CLS]` token embedding (768 dimensions) via `pooler_output`.
  - Dropout: Add `nn.Dropout(p=0.2)` to mitigate overfitting.
  - Classification Head: Linear projection layer `nn.Linear(768, 3)`.
  - Differential Learning Rate:
    - Backbone parameters (BERT): $lr = 1.5 \times 10^{-5}$
    - Head parameters: $lr = 5.0 \times 10^{-4}$
    - Weight decay: $0.01$
  - Learning Rate Scheduler: Linear Warmup (10% of total training steps) and Linear Decay.
  - Loss Function: `nn.CrossEntropyLoss(weight=class_weights)` with inverse frequency weights loaded directly from `features/train_holdout.json`.
- **Non-functional:**
  - Explicitly decouple parameter groups into `backbone_params` and `head_params`.
  - Support execution on both CPU and CUDA GPU environments.

## Architecture
```
input_ids, attention_mask
          │
          ▼
   bert-base-uncased (lr=1.5e-5)
          │
          ▼
   [CLS] vector (768d)
          │
          ▼
     Dropout(0.2)
          │
          ▼
Linear(768 -> 3) (lr=5e-4)
          │
          ▼
        Logits ──> CrossEntropyLoss(weight=class_weights)
```

## Related Code Files
- Modify: [`src/text.py`](../../src/text.py)
- Modify: [`src/models/text.py`](../../src/models/text.py)
- Reference: [`features/train_holdout.json`](../../features/train_holdout.json)

## Implementation Steps
1. Define class `BertMemeClassifier(nn.Module)`.
2. Add methods `backbone_params()` and `head_params()` to partition trainable parameters cleanly:
   ```python
   def head_params(self):
       return [p for n, p in self.named_parameters() if not n.startswith("bert.") and p.requires_grad]

   def backbone_params(self):
       return [p for n, p in self.named_parameters() if n.startswith("bert.") and p.requires_grad]
   ```
3. Build optimizer (`torch.optim.AdamW`) and warmup scheduler.
4. Load `class_weights` from `features/train_holdout.json` and initialize `CrossEntropyLoss`.
5. Run a forward pass test on a synthetic dummy batch to ensure output logits have shape `[batch_size, 3]` and scalar finite loss.

## Success Criteria
- [x] Logits tensor shape from forward pass is `[B, 3]`.
- [x] Backbone and head parameters are assigned distinct learning rates in AdamW.
- [x] Loss computation succeeds with `class_weights` without NaN or Inf values.
- [x] Gradients propagate properly upon calling `loss.backward()`.

## Risk Assessment
- **Risk:** Catastrophic forgetting if BERT learning rate is set too high.
  - *Mitigation:* Cap backbone learning rate at $1.5 \times 10^{-5}$.
- **Risk:** Majority class collapse due to heavy imbalance.
  - *Mitigation:* Ensure class-weighted loss is actively applied.
