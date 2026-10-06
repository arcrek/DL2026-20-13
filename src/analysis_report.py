"""Focused presentation of the Image, Text, or Both? research questions."""
from pathlib import Path

NAMES = {"text": "Text-only · BERT", "image": "Image-only · ResNet50",
         "both_concat": "Both · Concat", "both_cross_attn": "Both · Cross-Attention",
         "both_product": "Both · Product"}
SHORT = {"both_concat": "Concat", "both_cross_attn": "Cross-Attention", "both_product": "Product"}


def table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |"]
                     + ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows])


def render_report(experiment, summary, comparison, groups, group_summary, intervals, interactions,
                  transitions, errors, metadata, manifest, report_date, output, complementarity):
    overall = {r["model"]: r for r in summary}
    subgroup = {(r["group"], r["model"]): r for r in group_summary}
    paired = {(r["model_a"], r["model_b"]): r for r in comparison}
    contributions = {r["state"]: r for r in complementarity}
    concat = overall["both_concat"]
    baseline = overall["text"]
    best = max(summary, key=lambda r: r["macro_f1_mean"])
    concat_text = paired[("both_concat", "text")]
    pieces = []
    def add(text):
        pieces.append(text.strip())

    add("# Multimodal Meme Understanding: Image, Text, or Both?")
    add("## 1. Objective and comparison setup")
    add("We investigate how images, text, and their combination contribute to understanding a meme's message. "
        "In this experiment, sentiment classification measures one aspect of meme understanding: "
        "whether a meme is negative, neutral, or positive.")
    add("**Text-only (BERT)** receives only corrected text and is the main baseline. "
        "**Image-only (ResNet50)** receives the meme image and is the second baseline. "
        "**Both** receives the image and text, using three fusion methods: Concat, Cross-Attention, and Product. "
        "All models are evaluated on the same test memes.")
    add("**Macro-F1 is the primary metric:** it gives each sentiment class equal weight; higher is better. "
        "Accuracy is the percentage of correct predictions. `±` indicates the standard deviation across three seeds. "
        "We prioritize Macro-F1 because Positive accounts for 61.14% of the test set; "
        "favoring this majority class can produce a high Accuracy.")

    add("## 2. Which performs better: Image, Text, or Both?")
    add(table(["Model / input", "Macro-F1 mean ± SD", "Accuracy"],
              [(NAMES[r["model"]], f"{r['macro_f1_mean']:.4f} ± {r['macro_f1_sd']:.4f}",
                f"{100*r['accuracy_mean']:.2f}%") for r in summary]))
    add("![Macro-F1 comparison of Image, Text, and Both](figures/overall_performance.png)")
    add(f"Image-only and Text-only have similar Macro-F1 scores "
        f"({overall['image']['macro_f1_mean']:.4f} and {baseline['macro_f1_mean']:.4f}). "
        f"**{NAMES[best['model']]} achieves the highest mean Macro-F1**. "
        f"Concat improves by **{concat_text['delta_macro_f1']:+.4f}** over BERT, "
        f"but Accuracy drops from **{100*baseline['accuracy_mean']:.2f}%** to "
        f"**{100*concat['accuracy_mean']:.2f}%**. Combining both modalities therefore does not improve every metric.")

    add("## 3. Do images and text complement each other?")
    add("Similar average scores do not mean the two models succeed on the same memes. "
        "We compare their predictions for each test example:")
    add("![Cases where Image-only and Text-only make different correct predictions](figures/modality_complementarity.png)")
    add(f"On average per seed, **Text-only is correct on {contributions['text_only_correct']['n_mean_per_seed']:.1f} memes "
        "where Image-only is wrong**, while "
        f"**Image-only is correct on {contributions['image_only_correct']['n_mean_per_seed']:.1f} memes "
        "where Text-only is wrong**. These complementary predictions suggest an opportunity "
        "for Both to combine the strengths of each input.")
    add("The counts are averages across seeds, not 2,100 independent memes. "
        "This comparison does not identify which visual details or words explain the differences; "
        "meme images can also contain printed text.")

    add("## 4. When does Both help, and where does it fall short?")
    add(f"**By sentiment class:** Concat raises Neutral F1 from "
        f"**{baseline['class_f1_mean'][1]:.4f}** to **{concat['class_f1_mean'][1]:.4f}**, "
        f"but Positive F1 drops from **{baseline['class_f1_mean'][2]:.4f}** "
        f"to **{concat['class_f1_mean'][2]:.4f}**. Most of the overall Macro-F1 gain comes from "
        "better performance on Neutral; Both does not improve every sentiment class.")
    add("**By sarcasm:** text and images can convey conflicting messages, "
        "making sarcasm a useful subgroup for examining the benefits of combining both modalities.")
    add("![Performance on sarcastic and non-sarcastic memes](figures/sarcasm_subgroups.png)")
    non_delta = subgroup[("Non-sarcastic", "both_concat")]["macro_f1_mean"] - subgroup[("Non-sarcastic", "text")]["macro_f1_mean"]
    sar_delta = subgroup[("Sarcastic", "both_concat")]["macro_f1_mean"] - subgroup[("Sarcastic", "text")]["macro_f1_mean"]
    add(f"Concat improves over BERT by **{sar_delta:+.4f}** on sarcastic memes "
        f"({subgroup[('Sarcastic','text')]['n']} examples), and by **{non_delta:+.4f}** on non-sarcastic memes "
        f"({subgroup[('Non-sarcastic','text')]['n']} examples). "
        "The observed gain is smaller for sarcasm, and the confidence interval for the difference "
        "between these gains includes zero. "
        "**These results do not establish a special advantage of Both for sarcastic memes.**")

    add("## 5. Is there enough evidence to conclude that Both is better?")
    add("Not yet. The table compares each fusion method with the main baseline, BERT. "
        "A confidence interval (CI) that includes zero leaves the improvement uncertain. "
        "The Holm-adjusted p-value must be below 0.05 to meet the chosen significance threshold.")
    selected = [r for r in comparison if r["family"] == "primary" and r["model_b"] == "text"]
    add(table(["Fusion − BERT", "Macro-F1 gain", "95% CI", "Holm-adjusted p"],
              [(SHORT[r["model_a"]], f"{r['delta_macro_f1']:+.4f}",
                f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}]", f"{r['p_holm']:.4f}") for r in selected]))
    add("We use paired bootstrap and permutation procedures matched by meme ID, with 20,000 iterations each. "
        "All three seeds for a meme are kept together. Holm correction covers all six comparisons "
        "between the fusion methods and the two baselines; none meets the 0.05 threshold. "
        "These tests evaluate the saved trained models and do not fully capture variability from retraining.")

    add("## Conclusion")
    add("**Image and Text succeed on different examples, giving Both the potential to combine complementary signals. "
        "In this experiment, Concat achieves the highest mean Macro-F1, with the largest observed class-level "
        "improvement on Neutral. However, the gains are uneven, no special advantage for sarcasm is established, "
        "and the statistical evidence is insufficient to conclude that Both outperforms the baselines.**")
    add("No checkpoints are available for image/text masking experiments, so these results cannot yet "
        "be attributed to specific visual details or words.")
    add("**Supporting data:** [Full results and methods in JSON](analysis_summary.json), "
        "[confusion matrices](tables/confusion_matrices.csv), "
        "[20 error candidate IDs for Role F](tables/error_candidates.csv).\n\n"
        "Reproduce: `python -m src.analysis --report-date " + report_date + "`.")
    report = Path(output) / "role-e-analysis.md"
    report.write_text("\n\n".join(pieces) + "\n", encoding="utf-8")
    return report
