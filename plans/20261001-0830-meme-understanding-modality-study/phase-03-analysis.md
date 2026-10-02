# Phase 3: Analysis, ablations, errors
Files: `src/analysis.py`, `results/`, `report/figures/`
- [ ] Modality ablation at inference time (masking image or text representation).
- [ ] Fusion architecture comparison table (Concat vs Product vs Cross-Attention).
- [ ] Stratified analysis across meme dimensions: Performance on Sarcastic vs Non-sarcastic, Humorous vs Neutral memes.
- [ ] Permutation test as alternative; paired bootstrap significance testing for key claims (Multimodal vs Unimodal).
- [ ] Qualitative error analysis: inspect 20-30 misclassified memes, categorize root causes (visual metaphor, sarcasm/irony, cultural context, subtle OCR meaning).
- [ ] Generate publication-quality figures: comparison bar chart, confusion matrices, and representative error case panels.
Exit: all tables/figures exported to `report/figures/`.
Verify: every claim in the report maps directly to numerical evidence, confidence intervals, and statistical tests.
