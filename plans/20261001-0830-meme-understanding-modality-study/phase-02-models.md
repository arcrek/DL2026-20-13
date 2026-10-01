# Phase 2: Baselines and main models
Files: `src/heads.py`, `src/finetune.py`, `configs/*`, `results/*.json`
- [ ] Linear probes: CLIP-image only, CLIP-text only (3-class sentiment), across seeds 0-4.
- [ ] Multimodal fusion heads: concat linear, concat MLP, product MLP (rescaled), gated fusion. Same 5 seeds.
- [ ] Fine-tuned unimodal models: ViT-B/16 (image only) and BERT-base (text only), 3 seeds.
- [ ] Fine-tuned multimodal model: Partially unfrozen CLIP ViT-L/14 with Gated Fusion head.
- [ ] Log Macro F1-score, Accuracy, and per-class metrics to `results/`.
Verify: evaluate whether fusion outperforms unimodal image and text models; compute mean ± std across seeds.
Setup1 = Baseline vs Main models. Setup2 = Modality contribution (Image vs Text vs Both).
