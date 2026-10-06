---
phase: 3
title: "Training Loop & Contract Export (3 Seeds)"
status: completed
priority: P1
effort: "5h"
dependencies: ["1", "2"]
---

# Phase 3: Training Loop & Contract Export (3 Seeds)

## Overview
Execute end-to-end training and evaluation for the Text Baseline across 3 independent random seeds (`seed 0, 1, 2`), automatically selecting the optimal checkpoint according to hold-out Macro-F1, computing test split predictions, and exporting schema-compliant result files to `results/text_seed{0,1,2}.json`.

## Requirements
- **Functional:**
  - Standard PyTorch training loop with `epochs = 5`, `batch_size = 32`.
  - Automatic mixed precision (`torch.autocast("cuda", dtype=torch.bfloat16)`) for accelerated GPU computation.
  - Compute evaluation metrics after each epoch on `holdout`:
    - Macro-F1 (unweighted average of F1 across 3 classes).
    - Save `best_state_dict` corresponding to peak hold-out Macro-F1.
  - Upon completing training for each seed:
    - Reload `best_state_dict`.
    - Run inference on the `test` split (700 samples).
    - Collect: `ids`, `y_true`, `probs` (softmax matrix of shape `[700, 3]`), and `sarcasm`.
  - Invoke `save_results()` in [`src/results.py`](../../src/results.py) to export files:
    - Path: `results/text_seed{seed}.json`.
    - Validate contract via `validate_result(res)`.
  - Provide CLI flags: `--seeds 0 1 2`, `--epochs 5`, `--batch 32`, `--lr-backbone 1.5e-5`, `--lr-head 5e-4`.
- **Non-functional:**
  - Reproducibility: Seed all RNG sources (`random`, `np.random`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, DataLoader generator).
  - Strict evaluation integrity: The test split is evaluated only once after model selection.

## Architecture
```
For seed in [0, 1, 2]:
   Set all RNG seeds
   Train on fit_loader (5 epochs)
   Evaluate holdout_loader each epoch -> Track best holdout Macro-F1
   Load best model checkpoint
   Evaluate test_loader -> Softmax Probs
   Call save_results("text", seed, ids, y_true, probs, sarcasm, split="test")
   Output: results/text_seed{seed}.json
```

## Related Code Files
- Modify: [`src/text.py`](../../src/text.py)
- Modify: [`src/models/text.py`](../../src/models/text.py)
- Reference: [`src/results.py`](../../src/results.py)
- Reference: [`tests/test_results.py`](../../tests/test_results.py)
- Generate: `results/text_seed0.json`, `results/text_seed1.json`, `results/text_seed2.json`

## Implementation Steps
1. Implement `evaluate_split(model, dataloader, device)` returning `macro_f1`, `acc`, `probs`, `y_true`, `ids`, and `sarcasm`.
2. Implement `train_seed(seed, cfg, device)`:
   - Fix all random seeds.
   - Run training loop with gradient clipping (`clip_grad_norm_ <= 1.0`).
   - Track holdout Macro-F1 and checkpoint the best weights.
   - Run test evaluation and invoke `save_results()`.
3. Provide command-line interface entry point in `main()`.
4. Validate results against the test suite:
   ```bash
   pytest tests/test_results.py
   ```

## Success Criteria
- [x] Three files `results/text_seed0.json`, `results/text_seed1.json`, and `results/text_seed2.json` generated.
- [x] Function `validate_result()` passes with zero errors on all files.
- [x] Length of `ids`, `y_true`, `y_pred`, `probs`, `sarcasm` matches 700 samples.
- [x] Softmax probabilities are valid numbers in `[0, 1]` summing to 1.0.
- [x] `pytest tests/test_results.py` passes all assertions.

## Risk Assessment
- **Risk:** Training duration exceeding time budget on slower GPUs/CPUs.
  - *Mitigation:* Enable `torch.autocast`, use `batch_size=32`, and constrain training to 5 epochs.
- **Risk:** Missing required keys in results dictionary.
  - *Mitigation:* Collate and return `sarcasm` metadata from dataset records.
