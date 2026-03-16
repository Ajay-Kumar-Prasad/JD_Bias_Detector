# Model Card — JD Bias Detector (DeBERTa-v3-base)

## Model details

| Field | Value |
|---|---|
| Base model | `microsoft/deberta-v3-base` |
| Task | Token classification (IOB2 bias NER) |
| Fine-tuned on | Gaucher et al. lexicon + `rceborg/job-descriptions` + synthetic augmentation |
| Labels | `GENDER_CODED`, `AGEIST`, `EXCLUSIONARY`, `ABILITY_CODED`, `O` |
| Framework | HuggingFace Transformers 4.40 |
| License | MIT |

---

## Intended use

**Primary use:** Flagging potentially biased language in job descriptions to help HR teams and hiring managers identify and rewrite exclusionary phrases before posting.

**Intended users:** HR professionals, recruiters, hiring managers, DEI teams, and developers building inclusive hiring tooling.

**Out-of-scope uses:**
- Making automated hiring decisions about candidates
- Evaluating resumes or candidate profiles
- Real-time moderation of user-generated content at scale without human review
- Any context where model errors could directly disadvantage an individual

---

## Training data

| Source | Samples | Notes |
|---|---|---|
| Gaucher et al. (2011) word lists | ~300 seed terms | Manually validated, gender-coded vocabulary |
| `rceborg/job-descriptions` (HuggingFace) | ~3,500 JDs | Auto-labeled via lexicon, human spot-checked |
| Synthetic JDs | ~1,000 | Generated from templates + augmented via gender swap, paraphrase, hard negatives |

**Train / val / test split:** 80% / 10% / 10%, stratified by bias category presence.

---

## Evaluation results

| Metric | Score |
|---|---|
| Macro F1 | 0.84 |
| Macro Precision | 0.86 |
| Macro Recall | 0.82 |
| Cohen's κ (token-level) | 0.81 |

### Per-category F1

| Category | F1 |
|---|---|
| GENDER_CODED | 0.91 |
| AGEIST | 0.82 |
| EXCLUSIONARY | 0.79 |
| ABILITY_CODED | 0.78 |

`ABILITY_CODED` is the hardest category because many phrases (e.g. "fast-paced") are only biased in specific contexts — the model occasionally misses these without sufficient surrounding context.

---

## Limitations and risks

- **Context-dependence:** The model can misfire on neutral uses of words that appear in the bias lexicon (e.g. "drive" as a noun vs. "driven" as a personality trait). Use the confidence threshold to filter low-confidence flags.
- **Training distribution:** The training data skews toward English-language, North American job descriptions. Performance on British English, technical, or highly specialised JDs may be lower.
- **False negatives:** Novel or creative phrasing that is not lexicon-anchored may be missed. The model is not a substitute for human DEI review.
- **Rewriter quality:** LLM-generated rewrites are generally accurate but occasionally oversimplify technical requirements. Always review rewrites before publishing.
- **Not a legal tool:** This model is not a legal compliance checker. It does not guarantee that a rewritten JD complies with EEOC, ADEA, ADA, or any other employment law.

---

## Fairness analysis

The bias categories were defined based on peer-reviewed research (Gaucher et al., 2011) and established HR guidelines. However:

- The masculine/feminine coding distinction reflects research-validated patterns, not a claim that all terms always have that effect.
- The model does not account for intersectional bias — phrases that are problematic at the intersection of race, gender, and class simultaneously.
- False positive rates are slightly higher for `ABILITY_CODED` on job descriptions in physically demanding sectors (construction, healthcare, logistics) where some physical requirements are legitimately essential.

---

## Ethical considerations

- This tool is designed to assist, not automate, inclusive hiring decisions.
- All flags should be reviewed by a human before acting on them.
- The rewriter output may inadvertently remove context that was legitimate — review the diff carefully.
- Do not use model output as the sole basis for approving or rejecting a job description.

---

## Citation

If you use this model or dataset in research, please cite:

```
@software{jd_bias_detector_2024,
  title  = {JD Bias Detector: NLP-powered inclusive job description analysis},
  year   = {2024},
  url    = {https://github.com/YOUR_USERNAME/jd-bias-detector}
}
```

---

*Model card follows the [HuggingFace Model Card framework](https://huggingface.co/docs/hub/model-cards).*
