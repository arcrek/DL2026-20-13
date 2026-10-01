# Phase 1: Data and features
Files: `scripts/download_data.sh`, `DATA.md`, `src/data.py`, `src/features.py`
- [ ] Download Memotion 7k dataset (SemEval-2020 Task 8, `Ahren09/MMSoc_Memotion` on Hugging Face).
- [ ] Verify splits (`train`: 5593, `validation`: 699, `test`: 700) and 3-class sentiment label distribution (`negative`: 0, `neutral`: 1, `positive`: 2).
- [ ] Write `DATA.md`: source URL, paper citation, task definitions, splits, preprocessing, reproduction steps.
- [ ] Create a group-aware train hold-out (10% of train, grouped by caption text & image hash) for hyperparameter tuning.
- [ ] Extract and cache CLIP ViT-L/14 image and text embeddings for all splits to `.npy` (dim 768).
Verify: feature arrays have expected shapes `(N, 768)`; row counts match split sizes; sample IDs align with metadata.
