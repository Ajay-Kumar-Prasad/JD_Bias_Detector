# Model Card — JD Bias Detector (`deberta-jd-bias-v2-clean`)

## Model details

| Field | Value |
|---|---|
| Model name | `deberta-jd-bias-v2-clean` |
| Base model | `microsoft/deberta-v3-base` |
| Task | Token classification (IOB2 bias NER) |
| Labels | `GENDER_CODED`, `AGEIST`, `EXCLUSIONARY`, `ABILITY_CODED`, `O` |
| Framework | HuggingFace Transformers 4.40 |
| License | MIT |

---

## Intended use

**Primary use:** Detect potentially biased language in job descriptions and suggest neutral rewrites.

**Intended users:** Recruiters, hiring managers, HR/DEI teams, and developers building hiring tooling.

**Out-of-scope uses:**
- Automated candidate decisioning
- Resume or candidate profile scoring
- Legal compliance guarantees
- Fully automated moderation without human review

---

## Training configuration

- Epochs: `5`
- Loss: class-weighted cross-entropy
- Early stopping: enabled
- Output model: `models/deberta-jd-bias-v2-clean`

---

## Training data

| Source | Samples | Notes |
|---|---|---|
| Synthetic job descriptions | 2,769 | Generated from bias lexicons + templates + hard negatives |
| Public HuggingFace job descriptions | 409 | Lexicon-labeled and spot-checked |
| Bias lexicon seeds | ~300 terms | Used to construct and label spans |

**Total labeled JDs:** `3,178`

**Split sizes:**
- Train: `2,554`
- Val: `309`
- Test: `315`

**Data integrity:**
- Deduplicated across train/val/test
- No cross-split leakage (exact normalized text overlap = `0`)
- Biased vs neutral distribution is approximately balanced (~50/50)

---

## Evaluation results (held-out test set)

| Metric | Score |
|---|---|
| Macro F1 | **0.9612** |
| Precision | 0.9409 |
| Recall | 0.9824 |

### Per-category F1

| Category | F1 |
|---|---|
| GENDER_CODED | 0.9114 |
| AGEIST | 1.0000 |
| EXCLUSIONARY | 0.9901 |
| ABILITY_CODED | 0.9677 |

Full report: [`docs/evaluation_report.md`](evaluation_report.md)

---

## Limitations and risks

- Performance may drop on highly unstructured or domain-specific job descriptions.
- Some false positives occur from context ambiguity (e.g., "dynamic", "strong").
- Significant synthetic data component can introduce pattern bias.
- `AGEIST` is easier due to explicit phrasing and may not generalize to subtle cases.
- Rewriter output should always be reviewed by a human before publishing.

---

## Ethical considerations

- This tool is assistive, not a replacement for human hiring review.
- Output should not be used as the sole basis for inclusion/exclusion decisions.
- Bias labels reflect language-pattern risk, not legal determinations.

---

## Citation

```bibtex
@software{jd_bias_detector_2026,
  title  = {JD Bias Detector: NLP-powered inclusive job description analysis},
  year   = {2026}
}
```

---

*Model card follows the HuggingFace model card structure.*
