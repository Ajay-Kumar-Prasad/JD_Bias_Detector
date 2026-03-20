# Evaluation Report — JD Bias Detector

**Model:** `models/deberta-jd-bias-v2-clean`  
**Test samples:** 315  

---

## Overall Metrics

| Metric | Score |
|---|---|
| Macro F1 | **0.9612** |
| Macro Precision | 0.9409 |
| Macro Recall | 0.9824 |

---

## Per-category F1

| Category | F1 |
|---|---|
| GENDER_CODED | 0.9114 |
| AGEIST | 1.0000 |
| EXCLUSIONARY | 0.9901 |
| ABILITY_CODED | 0.9677 |

---

## Evaluation Notes

- Dataset is deduplicated across train/val/test splits.
- No cross-split leakage detected.
- Split balance is approximately 50/50 biased vs neutral.
- Metrics are computed on held-out test set (`n=315`).

---

## Seqeval-style Summary

```text
               precision    recall  f1-score   support

ABILITY_CODED     0.94      1.00      0.9677      46
AGEIST            1.00      1.00      1.0000      56
EXCLUSIONARY      0.98      1.00      0.9901      50
GENDER_CODED      0.84      1.00      0.9114      75

micro avg         0.94      0.98      0.9612     227
macro avg         0.94      0.98      0.9673     227
weighted avg      0.94      0.98      0.9608     227
```

---

## Error Analysis (Examples)

### False Positives

- "dynamic" flagged in neutral context
- "strong" flagged when used for technical proficiency

### False Negatives

- Subtle senior-leadership phrasing missed in low-context excerpts
- Indirect age preference phrasing not explicitly lexicalized

---

*Generated from the clean, deduplicated evaluation run.*
