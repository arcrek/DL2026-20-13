# Multimodal Meme Understanding: Image, Text, or Both?

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.0+-orange.svg)](https://huggingface.co/docs/transformers/index)
[![Tests](https://img.shields.io/badge/pytest-85%20passed-success.svg)](tests/)
[![Report](https://img.shields.io/badge/Report-16%20pages%20PDF-brightgreen.svg)](report/report.pdf)

> **Research Question:** Does combining visual and textual modalities improve meme sentiment classification over unimodal baselines, and is this gain statistically supported?

This repository contains the complete source code, training and evaluation pipelines, statistical analysis framework, Jupyter notebooks, interactive demo, and final LaTeX report for our empirical study on the **Memotion 7k** benchmark (SemEval-2020 Task 8).

---

## Table of Contents

- [Overview & Key Findings](#overview--key-findings)
- [Repository Structure](#repository-structure)
- [Installation & Environment Setup](#installation--environment-setup)
- [Dataset Preparation & Verification](#dataset-preparation--verification)
- [Quick Reproduction (One-Line)](#quick-reproduction-one-line)
- [Step-by-Step Reproduction Guide](#step-by-step-reproduction-guide)
  - [1. Data Preparation & Leakage-Free Holdout](#1-data-preparation--leakage-free-holdout)
  - [2. Model Training](#2-model-training)
  - [3. Evaluation & Statistical Significance Testing](#3-evaluation--statistical-significance-testing)
  - [4. Generating Figures & Confusion Matrices](#4-generating-figures--confusion-matrices)
- [Interactive Demo in Notebook](#interactive-demo-in-notebook)
- [Experimental Results Summary](#experimental-results-summary)
- [Testing](#testing)
- [Citation & Acknowledgments](#citation--acknowledgments)

---

## Overview & Key Findings

We evaluate five model configurations across three random seeds (`0`, `1`, `2`) on the identical 700-meme test partition:
1. **Text-only Baseline:** `bert-base-uncased` with pooler `[CLS]` token representation.
2. **Image-only Baseline:** `ResNet-50` with Global Average Pooling (GAP).
3. **Late Concatenation Fusion:** End-to-end concatenation of text `[CLS]` and image GAP vectors (`2,816` dims) with a 2-layer MLP head.
4. **Cross-Attention Fusion:** Unidirectional multi-head cross-attention (4 heads) where text queries ($L \times 256$) attend to $49$ spatial image feature vectors ($49 \times 256$) with residual LayerNorm.
5. **Product Fusion:** Element-wise Hadamard product of projected representations ($512$ dims) with LayerNorm and MLP classifier (ablation baseline).

**Key Conclusions:**
- **Uneven, Modest Gains:** Concatenation achieves the highest observed mean Macro-F1 (**33.69%**), gaining **+2.75 pp** over Text (**30.94%**) and **+2.87 pp** over Image (**30.81%**).
- **No Statistical Significance:** Rigorous paired bootstrap resampling (20,000 draws) yields 95% confidence intervals that cross zero for all fusion models against unimodal baselines (`[-0.11, +5.62]` for Concat vs. Text, adjusted Holm $p = 0.384$).
- **Prediction Shift Mechanism:** Fusion primarily shifts predictions toward the neutral class (+10.41 F1 points on neutral), which increases Macro-F1 under class imbalance, while slightly decreasing positive recall and accuracy.
- **Modality Complementarity:** Text-only and image-only models succeed on different subsets of memes (averaging 158.7 vs 145.7 unique correct predictions), yet fusion does not uniformly retain both successes.

---

## Repository Structure

```text
├── DATA.md                     # Dataset documentation, split distribution, and preprocessing rules
├── README.md                   # Project overview, installation, and reproduction guide
├── requirements.txt            # Python dependencies
├── configs/
│   └── base.yaml               # Canonical configuration (splits, seeds, class weights)
├── data/
│   └── memotion/               # Dataset directory (train.jsonl, validation.jsonl, test.jsonl, img/)
├── features/
│   └── train_holdout.json      # Leakage-free, group-aware 10% hold-out partition
├── notebooks/
│   └── meme_understanding.ipynb# Executable end-to-end notebook with training & Section 7 Demo
├── report/
│   ├── report.pdf              # Final compiled 16-page academic report
│   ├── report.tex              # LaTeX source code
│   ├── figures/                # Publication figures (confusion matrices, bootstrap distributions)
│   └── tables/                 # Exported statistical CSV tables
├── results/                    # 15 canonical test prediction JSONs (5 models x 3 seeds)
├── scripts/
│   ├── download_data.sh        # Download Memotion 7k dataset from Hugging Face
│   ├── make_confusion_matrix.py# Generate multi-seed confusion matrix grids & classification reports
│   └── run_all.sh              # One-command end-to-end verification and reproduction pipeline
├── src/
│   ├── analysis.py             # Reproducible statistical analysis, bootstrap & permutation tests
│   ├── config.py               # Path constants and configurations
│   ├── cross_attn.py           # Cross-attention CLI entry point
│   ├── data.py                 # Split verification, Union-Find grouping, holdout generator
│   ├── fusion.py               # Multimodal fusion CLI entry point
│   ├── image.py                # ResNet-50 image baseline CLI entry point
│   ├── models/                 # Neural network architectures
│   │   ├── cross_attention.py  # Text-to-image Cross-Attention implementation
│   │   ├── fusion.py           # Concat and Product fusion implementations
│   │   ├── image.py            # ResNet-50 image model implementation
│   │   └── text.py             # BERT text classifier implementation
│   ├── preprocessing.py        # Data cleaning, label normalization, image validation
│   ├── results.py              # Canonical result serialization and validation
│   └── text.py                 # BERT text baseline CLI entry point
└── tests/                      # 85 comprehensive pytest unit and regression tests
```

---

## Installation & Environment Setup

### Prerequisites
- Python `3.10`, `3.11`, or `3.12`
- (Optional) CUDA-compatible GPU for faster training (CPU execution is fully supported)

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/dl2026.git
cd dl2026
```

### 2. Set up virtual environment
Using `venv`:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt pytest
```

Or using [`uv`](https://github.com/astral-sh/uv) (recommended, faster):
```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt pytest
```

---

## Dataset Preparation & Verification

The project uses the **Memotion 7k** dataset (SemEval-2020 Task 8). Full specification, licensing, download links, split distributions, and preprocessing details are documented in [**`DATA.md`**](DATA.md).

### Download data
```bash
bash scripts/download_data.sh
```
This downloads and extracts `train.jsonl` (5,593 memes), `validation.jsonl` (699 memes), `test.jsonl` (700 memes), and all meme images into `data/memotion/img/`.

### Verify splits and integrity
Run the dataset verification suite:
```bash
python -m src.data --images
```
This validates:
- Expected split sizes (`5,593` / `699` / `700`)
- ID uniqueness and absence of cross-split contamination
- Image accessibility and corruption checks via `PIL.Image.verify()`
- Generates `features/train_holdout.json` using **Union-Find** grouping on MD5 image hashes and normalized captions to prevent template leakage between the fit set (`5,034` samples) and model-selection holdout (`559` samples).

---

## Quick Reproduction (One-Line)

To execute data verification, run the full 85-test regression suite, reproduce all statistical analyses from saved prediction files, and generate all publication figures in one command:

```bash
bash scripts/run_all.sh
```

---

## Step-by-Step Reproduction Guide

### 1. Data Preparation & Leakage-Free Holdout
Generate the deterministic, group-aware hold-out split and class-weighting schedule:
```bash
python -m src.data
```

### 2. Model Training
Each model is trained with seeds `0`, `1`, and `2`. Model checkpoints are chosen based on holdout Macro-F1, and test predictions are saved to `results/<config>_seed<seed>.json`.

- **Text Baseline (BERT):**
  ```bash
  python src/text.py --seeds 0 1 2 --epochs 5 --batch 32 --lr-backbone 1.5e-5 --lr-head 5e-4
  ```

- **Image Baseline (ResNet-50):**
  ```bash
  python src/image.py --seeds 0 1 2 --batch-size 32 --head-epochs 3 --finetune-epochs 5
  ```

- **Multimodal Fusion (Concat, Cross-Attention, Product):**
  ```bash
  # Train all 3 fusion architectures across 3 seeds:
  python src/fusion.py --config all --seeds 0 1 2 --epochs 5 --batch 16
  
  # Or train individual configurations:
  python src/fusion.py --config both_concat --seeds 0 1 2
  python src/fusion.py --config both_cross_attn --seeds 0 1 2
  python src/fusion.py --config both_product --seeds 0 1 2
  ```

### 3. Evaluation & Statistical Significance Testing
The pre-computed results for all 15 experiments are committed in `results/`. You can reproduce the full statistical analysis and hypothesis tests without retraining:

```bash
python -m src.analysis --report-date 2026-10-07
```
This calculates:
- Exact Macro-F1 and Accuracy per seed and mean $\pm$ standard deviation
- Per-class precision, recall, and F1 scores
- Subgroup evaluations (Sarcasm, Humor, Motivational)
- Paired bootstrap confidence intervals (20,000 resamples)
- Permutation tests with Holm-Bonferroni correction
- Complementary error transition matrices (corrected vs. induced errors)
- Outputs tabular CSV results into `report/tables/` and JSON summary into `report/analysis_summary.json`

### 4. Generating Figures & Confusion Matrices
Generate publication-quality confusion matrices and comparison charts:
```bash
python scripts/make_confusion_matrix.py
```
This generates:
- Per-model pooled confusion matrices: `report/figures/cm_*.png`
- Comparative grid figure: `report/figures/confusion_matrices_all.png` and vector PDF: `report/figures/confusion_matrices_all.pdf`
- Sarcasm-stratified confusion matrices: `report/figures/confusion_matrices_sarcasm.png`

---

## Interactive Demo in Notebook

The complete end-to-end workflow—from data loading to training and interactive inference—is available in **[`notebooks/meme_understanding.ipynb`](notebooks/meme_understanding.ipynb)**.

To run the interactive demo:
1. Open [`notebooks/meme_understanding.ipynb`](notebooks/meme_understanding.ipynb) in **Jupyter Lab**, **VS Code**, or **Google Colab**.
2. Scroll to **Section 7: Interactive Inference & Demo**.
3. Run the interactive cell:
```python
# Section 7 Interactive Demo
sample_image = 'data/memotion/img/test_00001.png'
sample_caption = 'THEY SAID I FIGHT LIKE A GIRL I TOOK THAT AS A COMPLIMENT'

pred_label, confidences, img_pil = predict_meme(
    image_path=sample_image,
    caption=sample_caption,
    model_type='both_concat'  # Options: 'both_concat', 'both_cross_attn', 'both_product', 'text', 'image'
)
```
4. The demo outputs:
   - Input meme image display
   - Predicted sentiment class (`NEGATIVE`, `NEUTRAL`, or `POSITIVE`)
   - Horizontal bar chart of class probabilities with exact confidence percentages.

---

## Experimental Results Summary

Performance evaluated on the 700-meme test partition across three random seeds (mean $\pm$ sample std):

| Model Architecture | Macro-F1 (%) | Accuracy (%) | Negative F1 (%) | Neutral F1 (%) | Positive F1 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Always Positive (Majority)** | 25.30 | **61.14** | 0.00 | 0.00 | 75.89 |
| **Text-only (BERT)** | 30.94 $\pm$ 2.25 | 45.71 $\pm$ 13.28 | 11.95 | 24.04 | 56.82 |
| **Image-only (ResNet-50)** | 30.81 $\pm$ 2.43 | 43.86 $\pm$ 6.01 | 7.50 | 26.18 | **58.77** |
| **Concat Fusion** | **33.69 $\pm$ 1.11** | 42.67 $\pm$ 2.44 | **13.49** | **34.45** | 53.11 |
| **Cross-Attention Fusion** | 32.69 $\pm$ 1.93 | 46.76 $\pm$ 6.62 | 8.53 | 31.14 | 58.38 |
| **Product Fusion (Ablation)**| 32.10 $\pm$ 0.19 | 40.86 $\pm$ 1.65 | 12.08 | 33.90 | 50.31 |

### Statistical Comparison (Fusion vs. Baselines)
Paired bootstrap test (20,000 draws, fixed class counts) and permutation test with Holm-Bonferroni correction:

| Comparison | $\Delta$ Macro-F1 (pp) | 95% Bootstrap CI (pp) | Holm-adjusted $p$ | Significant ($p < 0.05$)? |
| :--- | :---: | :---: | :---: | :---: |
| Concat vs. Text | +2.75 | `[-0.11, +5.62]` | 0.3840 | **No** |
| Concat vs. Image | +2.87 | `[-0.83, +6.56]` | 0.7112 | **No** |
| Cross-Attention vs. Text | +1.75 | `[-0.78, +4.25]` | 0.8288 | **No** |
| Cross-Attention vs. Image| +1.87 | `[-1.48, +5.14]` | 0.8288 | **No** |
| Product vs. Text | +1.16 | `[-1.68, +4.02]` | 0.9027 | **No** |
| Product vs. Image | +1.29 | `[-2.38, +4.95]` | 0.9027 | **No** |

---

## Testing

The test suite validates data preprocessing, label encoding, architecture shapes, loss functions, metrics calculation, and result schema contracts:

```bash
pytest tests/ -v
```

All 85 tests run offline (using CPU and mock fixtures where appropriate) and complete in under 20 seconds.

---

## Citation & Acknowledgments

The dataset is derived from SemEval-2020 Task 8:
```bibtex
@inproceedings{sharma2020semeval,
  title={{SemEval-2020 Task 8: Memotion Analysis--The Visuo-Lingual Metaphor!}},
  author={Sharma, Chhavi and Bhageria, Deepesh and Scott, William and PYKL, Srinivas and Das, Amitava and Chakraborty, Tanmoy and Pulabaigari, Viswanath and Gamb{\"a}ck, Bj{\"o}rn},
  booktitle={Proceedings of the Fourteenth Workshop on Semantic Evaluation},
  pages={759--773},
  year={2020}
}
```
