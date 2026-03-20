# JD Bias Detector

An NLP system for detecting biased language in job descriptions and rewriting it to be more inclusive.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow)](https://huggingface.co)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## Problem

Biased wording in job descriptions can reduce application rates from qualified candidates, especially from underrepresented groups. Manual review is slow and inconsistent.

## Solution

This project provides an end-to-end pipeline:

- Detects biased spans at token level
- Classifies bias type (`GENDER_CODED`, `AGEIST`, `EXCLUSIONARY`, `ABILITY_CODED`)
- Rewrites flagged phrases with neutral alternatives
- Returns inclusivity score + category breakdown
- Supports single and batch analysis through API and UI

---

## Model

- Base: `microsoft/deberta-v3-base`
- Task: Token Classification (IOB tagging)
- Training:
  - 5 epochs
  - Class-weighted loss
  - Early stopping
- Final Model: `deberta-jd-bias-v2-clean`

---

## Evaluation Results (Clean, Deduplicated Dataset)

- **Macro F1:** 0.9612  
- **Precision:** 0.9409  
- **Recall:** 0.9824  

### Per-category F1:
- GENDER_CODED: 0.9114  
- AGEIST: 1.0000  
- EXCLUSIONARY: 0.9901  
- ABILITY_CODED: 0.9677  

### Evaluation Notes

- Dataset is **deduplicated across train/val/test splits**
- **No cross-split leakage**
- Balanced biased vs neutral samples (≈50/50)
- Evaluation performed on held-out test set (315 samples)

📄 Full evaluation: [`docs/evaluation_report.md`](docs/evaluation_report.md)

---

## Dataset

The dataset is constructed from:

- Synthetic job descriptions generated using bias lexicons
- Public job description datasets (HuggingFace)
- Post-processing with:
  - Deduplication
  - Balanced splits
  - IOB token labeling

> ⚠️ Note: Dataset includes a significant synthetic component.  
> Real-world generalization is validated separately via manual testing.

Detailed schema and split stats: [`data/README.md`](data/README.md)

---

## Limitations

- Performance may drop on highly unstructured or domain-specific job descriptions
- Some false positives occur due to context ambiguity (e.g., "dynamic", "strong")
- Synthetic data may introduce pattern bias
- AGEIST category is easier due to explicit phrasing, may not generalize to subtle cases

---

## Model Usage

```python
from transformers import pipeline

ner = pipeline(
    "token-classification",
    model="path/to/deberta-jd-bias-v2-clean",
    aggregation_strategy="simple"
)

ner("Looking for a young, hungry developer")
```

---

## API Usage

### Analyze one JD

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-secret-key" \
  -d '{
    "text": "We are looking for a rockstar engineer who is young and hungry to crush it."
  }'
```

### Batch analyze JDs

```bash
curl -X POST http://localhost:8000/analyze/batch \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-secret-key" \
  -d '{
    "texts": [
      "We are looking for a rockstar engineer.",
      "We value collaboration and clear communication."
    ]
  }'
```

---

## Deployment

### Local

```bash
pip install -r requirements.txt
cp .env.example .env
uvicorn api.main:app --reload
streamlit run app/streamlit_app.py
```

### Docker

```bash
docker-compose up --build
```

---

## Key Improvements

- Removed cross-split data leakage
- Implemented dataset deduplication
- Balanced biased vs neutral samples
- Added class-weighted training
- Improved generalization on unseen job descriptions

---

## Training / Re-evaluation

```bash
python -m training.data_prep --output_dir data/annotated --source both --n_synthetic 1200
python -m training.train --config training/configs/deberta_base.yaml
python -m training.evaluate --model_dir models/deberta-jd-bias-v2-clean --test_data data/annotated/test.jsonl --output docs/evaluation_report.md
```

---

## Repository Structure

```text
api/        FastAPI backend (analyze, batch analyze, health, metrics)
app/        Streamlit frontend + batch CSV analyzer
data/       Annotated data + bias lexicons + data documentation
training/   Data prep, training, evaluation, notebooks
docs/       Evaluation report, model card, API docs
```

---

## License

MIT — see [LICENSE](LICENSE).
