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
| `train` | 5,593 | ~1,000 | ~2,100 | ~2,493 |
| `validation` | 699 | ~125 | ~260 | ~314 |
| `test` | 700 | ~125 | ~260 | ~315 |
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

## 4. Preprocessing & Feature Extraction
- **Group-Aware Hold-Out:** A stratified 10% subset (~560 samples) of `train` is held out for hyperparameter tuning and early stopping. Groups are formed using Union-Find on normalized captions and image MD5 hashes to guarantee that near-duplicate or identical memes do not leak between the training fit set and the hold-out set.
- **Feature Representations:**
  - Backbone: `openai/clip-vit-large-patch14`.
  - Image: Extracted via `CLIPVisionModel` and projected through `visual_projection` to a 768-dimensional L2-normalized vector.
  - Text: Tokenized with `CLIPTokenizer` (max 77 tokens), passed through `CLIPTextModel`, and projected through `text_projection` to a 768-dimensional L2-normalized vector.
  - Outputs cached per split as `{split}_img.npy`, `{split}_txt.npy`, `{split}_labels.npy`, and `{split}_ids.npy`.

## 5. Reproduction Instructions
```bash
# 1. Download and unpack dataset
bash scripts/download_data.sh

# 2. Verify splits, build group-aware hold-out
python src/data.py

# 3. Extract and cache CLIP ViT-L/14 embeddings (GPU)
python src/features.py
```
Or run the complete pipeline directly via `notebooks/meme_understanding.ipynb` on Google Colab.
