# Phase 1: Data and features (Hours 0-6)
Files: `scripts/download_data.sh`, `DATA.md`, `src/data.py`, `tests/test_data.py`, `features/train_holdout.json`
Roadmap: [docs/project-roadmap.md](../../docs/project-roadmap.md) §Phase 1

- [x] Download Memotion 7k (SemEval-2020 Task 8, `Ahren09/MMSoc_Memotion`).
- [x] Switch text pipeline from `text_ocr` to `text_corrected` everywhere (fix sample-notebook bug); assert no code path reads `text_ocr`.
- [x] Verify splits (train 5593 / val 699 / test 700) and 3-class labels (neg 0, neu 1, pos 2).
- [x] Union-Find grouping in `src/data.py` (image MD5 + normalized text) so no template is shared between fit and hold-out; 10% group-aware hold-out -> `features/train_holdout.json`.
- [x] Compute `class_weights` for Negative (~9%), Neutral (~31%), Positive (~59%); actual train counts 518/1762/3313.
- [x] Write `DATA.md`: source, citation, splits, preprocessing, leakage control, reproduction steps.

Verify (mechanical):
- `pytest tests/test_data.py` passes.
- `features/train_holdout.json` exists, valid JSON, zero group overlap between fit and hold-out (test asserts it).
