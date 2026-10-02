# Phase 2: Three controlled models (Hours 6-18)
Files: `src/finetune.py`, `results/*.json`
Roadmap: [docs/project-roadmap.md](../../docs/project-roadmap.md) §Phase 2
Protocol: identical hold-out split, class-weighted loss, seeds 0, 1, 2 for every config.

- [ ] Branch 1, Text-only: `bert-base-uncased` [CLS] 768d -> Dropout -> Linear(3); differential LR 1.5e-5 (BERT) / 5e-4 (head).
- [ ] Branch 2, Image-only: ResNet50 (VGG16 fallback) GAP 2048d -> Dropout -> Linear(3); 2-stage transfer (freeze backbone, then unfreeze fine-tune).
- [ ] Branch 3A, Both / Late Concat: `[v_text, v_img]` -> MLP classifier.
- [ ] Branch 3B, Both / Cross-Attention: Query = text tokens (L x 256), Key/Value = image feature map (49 x 256) -> residual LayerNorm -> MLP.
- [ ] Ablation config, Product Fusion: elementwise product of `[v_text, v_img]` -> MLP (for Concat vs Product vs Cross-Attention comparison in Phase 3).
- [ ] Save per-seed predictions and Accuracy / Macro-F1 to `results/*.json`; aggregate table `mean ± std` for Text, Image, Both.

Verify (mechanical):
- `results/` contains one JSON per (config, seed) with keys `accuracy`, `macro_f1`, `predictions`.
- Aggregate table exists with 3 seeds per config; Text/Image/Both rows all populated.
