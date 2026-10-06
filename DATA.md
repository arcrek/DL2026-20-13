# Dataset Documentation: Memotion 7k (SemEval-2020 Task 8)

This document provides complete specification for the dataset used across all experiments in this repository: official dataset URLs, versioning, data partition splits, preprocessing procedures, leakage prevention, and executable scripts required to reproduce the data.

---

## 1. Dataset Overview & Official URLs

- **Benchmark Name:** Memotion Dataset 7k
- **Official Task:** SemEval-2020 Task 8: *Memotion Analysis — The Evaluation of Visuo-Lingual Metaphors in Memes* (Task A: Sentiment Analysis).
- **Official Publication:**
  - Sharma et al., *"SemEval-2020 Task 8: Memotion Analysis–The Visuo-Lingual Metaphor!"*, Proceedings of the Fourteenth Workshop on Semantic Evaluation (SemEval 2020), pages 759–773 ([arXiv:2008.03781](https://arxiv.org/abs/2008.03781)).
- **Official Dataset URLs & Repositories:**
  - **Primary Author Release (Kaggle):** [https://www.kaggle.com/datasets/williamscott/memotion-dataset-7k](https://www.kaggle.com/datasets/williamscott/memotion-dataset-7k)
  - **Official CodaLab Competition:** [SemEval-2020 Task 8 on CodaLab](https://competitions.codalab.org/competitions/20643)
  - **Hugging Face Mirror (Used in this repository):** [`Ahren09/MMSoc_Memotion`](https://huggingface.co/datasets/Ahren09/MMSoc_Memotion)
- **Dataset Version & Snapshot Hash:**
  - Tracked metadata Git commit hash: `cdb15b61d84d56db73e0e59535dfea81ea3c22f4`
  - Release version: Memotion 1.0
- **License & Domain:** Academic research and educational evaluation on general internet memes annotated for sentiment, humor, sarcasm, and motivational intent.

---

## 2. Data Splits and Distributions

The dataset is partitioned into official standard splits stored in `data/memotion/`:
- `train.jsonl` (5,593 memes)
- `validation.jsonl` (699 memes)
- `test.jsonl` (700 memes)

### Class Distribution (3 Target Classes)

The sentiment distribution is heavily skewed toward positive memes (~60%), with negative memes representing less than 10%:

| Split | Number of Memes | Negative (Class 0) | Neutral (Class 1) | Positive (Class 2) |
|---|:---:|:---:|:---:|:---:|
| `train` | 5,593 | 518 (9.3%) | 1,762 (31.5%) | 3,313 (59.2%) |
| `validation` | 699 | 59 (8.4%) | 221 (31.6%) | 419 (59.9%) |
| `test` | 700 | 54 (7.7%) | 218 (31.1%) | 428 (61.1%) |
| **Total** | **6,992** | **631 (9.0%)** | **2,201 (31.5%)** | **4,160 (59.5%)** |

### Leakage-Free Holdout Partition (`train_holdout.json`)
For hyperparameter tuning and model checkpoint selection (early stopping), `train` is partitioned into:
- **`fit` subset:** 5,034 samples used for parameter optimization.
- **`holdout` subset:** 559 samples (10% stratified holdout) used exclusively for checkpoint selection.

**Leakage Prevention Guarantee:**
Memes frequently share the same underlying visual template with altered text captions, or identical captions on slightly cropped images. To prevent data leakage between `fit` and `holdout`:
- A **Union-Find connected components algorithm** (`src/data.py:group_ids`) groups memes sharing either identical MD5 image hashes or identical normalized captions.
- Groups are kept atomic: no meme group is split across `fit` and `holdout`.

### Class-Weighted Loss Schedule
Because positive samples outnumber negative samples by more than 6:1, inverse-frequency class weights $w_c$ are calculated over the 5,034 fit samples:

$$
w_c = \frac{N_{\text{fit}}}{C \cdot N_{c, \text{fit}}}
$$

where $C = 3$ is the number of classes. The resulting weights applied during PyTorch cross-entropy loss are:
- `Negative (0):` $\approx 3.66$
- `Neutral (1):` $\approx 1.06$
- `Positive (2):` $\approx 0.56$

Stored in `features/train_holdout.json` and consumed automatically by all training modules.

---

## 3. Preprocessing Procedure

The preprocessing pipeline is implemented in [`src/preprocessing.py`](src/preprocessing.py) and consists of three deterministic steps:

### Step 1: Label Consolidation & Field Organization
Raw Memotion annotations classify sentiment across a 5-point scale (`very_positive`, `positive`, `neutral`, `negative`, `very_negative`). These are consolidated into 3 target classes:

$$
y_{\text{norm}} = \begin{cases}
\text{positive}, & y_{\text{raw}} \in \{\text{very positive}, \text{positive}, \text{pos}\} \\
\text{neutral},  & y_{\text{raw}} \in \{\text{neutral}, \text{neu}\} \\
\text{negative}, & y_{\text{raw}} \in \{\text{very negative}, \text{negative}, \text{neg}\}
\end{cases}
$$

| Target Class (Normalized) | Integer ID | Raw SemEval-2020 Labels |
|---|:---:|---|
| `positive` | 2 | `very_positive`, `positive`, `pos` |
| `neutral`  | 1 | `neutral`, `neu` |
| `negative` | 0 | `very_negative`, `negative`, `neg` |

**Text Selection Rule:**
Only human-corrected text captions (`text_corrected`, saved as `text`) are used. Raw OCR output (`text_ocr`) contains high character error rates, merged words, and layout artifacts that degrade model representations. `tests/test_data.py` asserts that `text_ocr` is never consumed by training loaders.

### Step 2: Image Verification & Corruption Filtering
During batch training with PyTorch `DataLoader`, corrupted or truncated image files cause unhandled `OSError: image file is truncated`.
- The preprocessing function `is_valid_image(path)` verifies:
  1. `PIL.Image.verify()` passes without error.
  2. The image can be converted to 3-channel RGB: `img.convert("RGB")`.
- Truncated files (such as `train_04578.png`) are filtered out prior to training.
- In addition, PyTorch dataset loaders enable `ImageFile.LOAD_TRUNCATED_IMAGES = True` and implement a fallback white tensor mechanism (`safe_open_image`) as defense-in-depth.

### Step 3: Numeric Label Encoding
Standardized integer IDs are mapped for loss computation:
```python
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
```

---

## 4. Metadata Fields

Each sample in `data/memotion/*.jsonl` contains the following fields:

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Unique sample identifier (e.g. `"train_00001"`, `"test_00000"`) |
| `img` | `str` | Relative path to image file (e.g. `"img/train_00001.png"`) |
| `text` | `str` | Human-corrected caption embedded in the meme |
| `label` | `int` | Integer sentiment class (`0`: negative, `1`: neutral, `2`: positive) |
| `raw_sentiment` | `str` | Original 5-scale sentiment label from SemEval |
| `humor` | `str` | Humor degree (`not_funny`, `funny`, `very_funny`, `hilarious`) |
| `sarcasm` | `str` | Sarcasm degree (`not_sarcastic`, `general`, `twisted_meaning`, `very_twisted`) |
| `motivational` | `str` | Motivational tone (`motivational`, `not_motivational`) |

---

## 5. Scripts Required to Reproduce Data

All scripts needed to download, verify, preprocess, and partition the dataset are included in this repository:

### 1. Download Dataset
Downloads the raw parquet files from Hugging Face and extracts images and JSONL files into `data/memotion/`:
```bash
bash scripts/download_data.sh
```

### 2. Verify Splits & Generate Leakage-Free Holdout
Verifies all 6,992 sample counts, checks ID uniqueness, ensures zero cross-split overlap, tests image integrity, and writes `features/train_holdout.json`:
```bash
python -m src.data --images
```

### 3. Run Preprocessing Pipeline
Executes the full 3-step cleaning, image verification, and label encoding pipeline on the local dataset:
```bash
python -m src.preprocessing
```

### 4. Run Automated Data Contract Tests
Executes the pytest test suite to ensure dataset integrity, holdout properties, and preprocessing compliance:
```bash
pytest tests/test_data.py tests/test_preprocessing.py -v
```
All tests must pass (`9 passed`).
