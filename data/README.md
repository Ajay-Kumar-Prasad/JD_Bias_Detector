# Data README

## Contents

- `annotated/train.jsonl`
- `annotated/val.jsonl`
- `annotated/test.jsonl`
- `bias_lexicon/*.json`

Each JSONL row follows:

```json
{
  "tokens": ["We", "are", "looking", "for", "..."],
  "labels": ["O", "O", "O", "O", "..."],
  "text": "We are looking for ...",
  "source": "synthetic_biased"
}
```

## Label schema

IOB2 tags:

- `B-GENDER_CODED`, `I-GENDER_CODED`
- `B-AGEIST`, `I-AGEIST`
- `B-EXCLUSIONARY`, `I-EXCLUSIONARY`
- `B-ABILITY_CODED`, `I-ABILITY_CODED`
- `O`

## Current split stats

- Total samples: `3178`
- Train: `2554`
- Val: `309`
- Test: `315`

Split overlap (exact normalized text) is checked and currently:

- `train ∩ val = 0`
- `train ∩ test = 0`
- `val ∩ test = 0`

## Generation / refresh

Regenerate splits with leakage checks:

```bash
python -m training.data_prep \
  --output_dir data/annotated \
  --source both \
  --n_synthetic 1200
```

`training.data_prep` now de-duplicates by normalized text and enforces cross-split uniqueness before save.
