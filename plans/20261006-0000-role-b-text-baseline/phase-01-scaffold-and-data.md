---
phase: 1
title: "Module Scaffold & Data Pipeline"
status: completed
priority: P1
effort: "2h"
dependencies: []
---

# Phase 1: Module Scaffold & Data Pipeline

## Overview
Initialize an independent module structure for Role B at `src/text.py` and `src/models/text.py` to ensure modularity and prevent interference with other roles (Role C `image.py`, Role D `fusion.py`), while establishing a text loading pipeline that strictly avoids meme template leakage.

## Requirements
- **Functional:**
  - Ingest data from `data/memotion/*.jsonl` via `load_jsonl()` in [`src/data.py`](../../src/data.py).
  - Partition training data into `fit` and `holdout` based on [`features/train_holdout.json`](../../features/train_holdout.json).
  - Extract 3-class target labels: Negative (0), Neutral (1), Positive (2).
  - Extract `sarcasm` metadata tags to support downstream error analysis by Role E.
  - Employ `AutoTokenizer.from_pretrained("bert-base-uncased")` with `padding=True`, `truncation=True`, `max_length=128`.
- **Non-functional:**
  - **Data Hygiene & Leakage Control:** Never ingest the `text_ocr` field; use only `text` (extracted from `text_corrected`).
  - **Code Isolation:** Avoid editing [`src/finetune.py`](../../src/finetune.py) directly to prevent merge conflicts across teammates.

## Architecture
```
data/memotion/train.jsonl ──┐
                            ├──> TextMemeDataset ──> DataLoader(batch=32, shuffle=True)
features/train_holdout.json ┘            │
                                         ▼
                            Tokenizer('bert-base-uncased')
                               (max_len=128, return_pt)
                                         │
                                         ▼
                             Batch: {ids, input_ids, mask, y, sarcasm}
```

## Related Code Files
- Create: [`src/text.py`](../../src/text.py)
- Create: [`src/models/text.py`](../../src/models/text.py)
- Reference: [`src/data.py`](../../src/data.py)
- Reference: [`configs/base.yaml`](../../configs/base.yaml)

## Implementation Steps
1. Create `src/models/` directory and `src/models/__init__.py`.
2. Define class `TextMemeDataset(Dataset)` to load `id`, `text`, `label`, and `sarcasm`.
3. Implement `make_collate_fn(tokenizer, max_length=128)` for batch gathering, dynamic padding, and truncation into PyTorch tensors.
4. Set up DataLoader instances for `fit`, `holdout`, and `test` with `torch.Generator` for deterministic batching.
5. Run a sanity check: sample 1 batch from `fit_loader`, verify `input_ids` and `attention_mask` shapes (`[batch_size, seq_len]`), and inspect `labels`.

## Success Criteria
- [x] Files `src/text.py` and `src/models/text.py` created and importable.
- [x] Generated batches contain all required keys: `ids`, `input_ids`, `attention_mask`, `labels`, `sarcasms`.
- [x] The `fit` split contains 5,034 samples, and `holdout` contains 559 samples (summing to 5,593 training samples).
- [x] Sanity check executes cleanly without tokenizer warnings or errors.

## Risk Assessment
- **Risk:** Slow tokenization or network timeout when downloading from Hugging Face Hub.
  - *Mitigation:* Cache pretrained weights locally.
- **Risk:** Out of Memory (OOM) caused by overly long meme text.
  - *Mitigation:* Enforce `truncation=True, max_length=128`.
