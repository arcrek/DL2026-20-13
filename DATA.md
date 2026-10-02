# DATA: Memotion 7k (SemEval-2020 Task 8)

## 1. Dataset Overview
- **Name:** Memotion Dataset 7k
- **Benchmark / Task:** SemEval-2020 Task 8: *Memotion Analysis — The Evaluation of Visuo-Lingual Metaphors in Memes* (Sharma et al., SemEval 2020, [arXiv:2008.03781](https://arxiv.org/abs/2008.03781)).
- **Repository Used:** Hugging Face [`Ahren09/MMSoc_Memotion`](https://huggingface.co/datasets/Ahren09/MMSoc_Memotion).
- **Domain:** General internet memes annotated for sentiment, humor, sarcasm, offensiveness, and motivational intent.
- **License / Usage:** Academic research and educational evaluation.

## 2. Splits and Distributions
The dataset is partitioned into official standard splits:

| Split | Number of Memes | Negative (0) | Neutral (1) | Positive (2) |
|---|---|---|---|---|
| `train` | 5,593 | 518 | 1,762 | 3,313 |
| `validation` | 699 | 59 | 221 | 419 |
| `test` | 700 | 54 | 218 | 428 |
| **Total** | **6,992** | | | |

*Note: In SemEval-2020 Task A, the 5-point sentiment scale (`very_negative`, `negative`, `neutral`, `positive`, `very_positive`) is consolidated into 3 standard classes: Negative (0), Neutral (1), and Positive (2).*

## 3. Annotations and Metadata
Each sample contains:
- `image`: The meme image.
- `text_corrected`: Extracted and human-corrected textual caption embedded in the meme.
- `sentiment`: Target sentiment class (`negative`, `neutral`, `positive`).
- Rhetorical & emotional context:
  - `humor`: `not_funny`, `funny`, `very_funny`, `hilarious`
  - `sarcasm`: `not_sarcastic`, `general`, `twisted_meaning`, `very_twisted`
  - `motivational`: `motivational`, `not_motivational`
  - `offensive`: `not_offensive`, `slight`, `very_offensive`, `hateful_expressive` (general internet offensiveness scale, used strictly as qualitative metadata).

## 4. Preprocessing
- **Group-Aware Hold-Out:** A stratified 10% subset (~560 samples) of `train` is held out for hyperparameter tuning and early stopping. Groups are formed using Union-Find on normalized captions and image MD5 hashes to guarantee that near-duplicate or identical memes do not leak between the training fit set and the hold-out set.

- **Text field:** Only `text_corrected` (human-corrected) is used, stored as `text` in the local JSONL. `text_ocr` is never read; `tests/test_data.py` asserts this.
- **Class weights:** Inverse-frequency, `n / (3 * count_c)`, computed on the fit subset and stored in `features/train_holdout.json` (`class_weights`). Train is imbalanced (~9% / 31% / 59%).
- **Leakage control:** Groups merge memes sharing a normalized caption or an identical image (MD5). A whole group lands in either fit or hold-out.

## 5. Reproduction Instructions
```bash
# 1. Download and unpack dataset
bash scripts/download_data.sh

# 2. Verify splits, build group-aware hold-out
python3 src/data.py
```
Or run the complete pipeline directly via `notebooks/meme_understanding.ipynb` on Google Colab.
