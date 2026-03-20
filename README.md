# JD Bias Detector
A production-oriented NLP service that detects biased language in job descriptions, explains why it is problematic, and generates inclusive rewrites with confidence-aware rewrite modes.

## 1. Problem Statement
Hiring teams often rely on manual review to identify biased wording in job descriptions. That process is slow, inconsistent, and difficult to scale across high-volume recruiting workflows. Biased phrasing can reduce application rates from qualified candidates and weaken diversity efforts.

## 2. Solution Overview
JD Bias Detector provides an end-to-end pipeline:
- Detects biased spans at token level
- Classifies each span into one of four bias categories
- Generates rewrite suggestions with explanations
- Applies confidence-aware rewrite policy (`auto_replace`, `suggest`, `ignore`)
- Computes an inclusivity score and per-category breakdown
- Exposes functionality via FastAPI and a Streamlit UI (single + batch mode)

## 3. Key Features
- Token-level bias detection using fine-tuned DeBERTa-v3-base
- Four supported categories: `GENDER_CODED`, `AGEIST`, `EXCLUSIONARY`, `ABILITY_CODED`
- Confidence policy controls for rewrite behavior
- Configurable thresholds at runtime from the UI
- API key protection for `/analyze/*` endpoints
- Batch CSV analysis and downloadable output in Streamlit
- Docker Compose setup for API + UI

## 4. System Architecture
```text
Client (Recruiter / Analyst)
        |
        v
Streamlit UI (app/streamlit_app.py)
        |
        v
FastAPI Backend (api/main.py)
  ├─ Classifier (DeBERTa token classification)
  ├─ Rewriter (Anthropic API or deterministic fallback templates)
  └─ Scorer (weighted penalty -> inclusivity score)
        |
        v
JSON response (spans, rewrite modes, score, rewritten text)
```

## 5. ML / NLP Pipeline
1. Input JD text is validated (`min_length=10`).
2. Token classifier predicts biased entities and confidence scores.
3. Classifier post-processing merges adjacent spans and filters low-signal tokens.
4. Rewriter enriches each span with:
   - `rewrite`
   - `explanation`
5. Rewrite policy assigns each span to:
   - `auto_replace` (high confidence)
   - `suggest` (medium confidence)
   - `ignore` (low confidence)
6. `auto_replace` spans are applied to produce `rewritten_text`.
7. Scorer computes inclusivity score (0–100) and category counts.

## 6. Model Details
- **Base model:** `microsoft/deberta-v3-base`
- **Task:** Token classification (IOB2 labels)
- **Label set:**
  - `B-GENDER_CODED`, `I-GENDER_CODED`
  - `B-AGEIST`, `I-AGEIST`
  - `B-EXCLUSIONARY`, `I-EXCLUSIONARY`
  - `B-ABILITY_CODED`, `I-ABILITY_CODED`
  - `O`
- **Training approach (implemented):**
  - Weighted cross-entropy loss (class imbalance handling)
  - Early stopping (`early_stopping_patience: 2`)
  - Linear LR schedule, warmup, gradient accumulation
  - Validation each epoch; best model checkpoint retained
- **Inference behavior:**
  - Confidence thresholds and stop-word filtering
  - Optional temperature scaling (`CLASSIFIER_CALIBRATION_TEMPERATURE`)

## 7. Evaluation
Metrics below are from `docs/evaluation_report.md` (held-out test set, `n=315`):

| Metric | Score |
|---|---:|
| Macro F1 | **0.9612** |
| Macro Precision | 0.9409 |
| Macro Recall | 0.9824 |

Per-category F1:

| Category | F1 |
|---|---:|
| GENDER_CODED | 0.9114 |
| AGEIST | 1.0000 |
| EXCLUSIONARY | 0.9901 |
| ABILITY_CODED | 0.9677 |

Evaluation data notes in repo:
- Deduplicated train/val/test splits
- No cross-split overlap by normalized text
- Approximate 50/50 biased vs neutral split

## 8. Project Structure
```text
api/
  main.py                 FastAPI app, routing, startup lifecycle
  routes/
    analyze.py            Single + batch analysis endpoints
    health.py             Health and metrics endpoints
  models/
    classifier.py         DeBERTa token-classification wrapper
    rewriter.py           LLM + fallback rewrite generation
    scorer.py             Inclusivity score computation
  schemas.py              Request/response contracts

app/
  streamlit_app.py        Streamlit UI (single + batch modes)
  components/             Sidebar, highlighting, diff rendering
  assets/style.css        UI styling

training/
  data_prep.py            Dataset generation and split pipeline
  train.py                Model fine-tuning entrypoint
  evaluate.py             Evaluation and markdown report generation
  configs/deberta_base.yaml

data/
  annotated/              train/val/test JSONL
  bias_lexicon/           Bias lexicon sources

docs/
  evaluation_report.md    Current evaluation report
  api_reference.md        API docs
  model_card.md           Model documentation

tests/
  test_api.py             API integration tests (mocked models)
  test_classifier.py      Classifier unit tests
```

## 9. Quickstart
### Prerequisites
- Python 3.10+
- `pip`

### Setup
```bash
git clone <repo-url>
cd JD_Bias_Detector
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### Run API
```bash
uvicorn api.main:app --reload --port 8001
```

### Run Streamlit UI (new terminal)
```bash
streamlit run app/streamlit_app.py
```

### Useful local URLs
- API docs: `http://127.0.0.1:8001/docs`
- Health: `http://127.0.0.1:8001/health`
- Streamlit: `http://localhost:8501`

### Docker Compose
```bash
docker-compose up --build
```

## 10. API Usage
`/analyze/*` requires `x-api-key`.

### Request
```bash
curl -X POST http://127.0.0.1:8001/analyze/ \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-secret-key" \
  -d '{
    "text": "We are looking for a rockstar engineer who is young and hungry to crush it in a fast-paced environment.",
    "auto_rewrite_threshold": 0.85,
    "suggestion_threshold": 0.70
  }'
```

### Response (shape)
```json
{
  "inclusivity_score": 72,
  "flagged_spans": [
    {
      "text": "rockstar engineer",
      "start": 26,
      "end": 43,
      "category": "EXCLUSIONARY",
      "confidence": 0.80,
      "rewrite_confidence": 0.80,
      "rewrite_mode": "suggest",
      "explanation": "This phrase uses exclusionary jargon that may deter underrepresented candidates.",
      "rewrite": "highly skilled engineer"
    }
  ],
  "rewritten_text": "...",
  "category_breakdown": {
    "GENDER_CODED": 1,
    "AGEIST": 1,
    "EXCLUSIONARY": 1,
    "ABILITY_CODED": 1
  }
}
```

## 11. Demo / UI
The Streamlit app supports:
- Single JD analysis with highlighted spans
- Confidence controls:
  - display threshold
  - auto-rewrite threshold
  - suggestion threshold
- Side-by-side original vs rewritten diff
- Rewrite mode summary (`auto-replaced`, `suggested`, `ignored`)
- Batch CSV upload and downloadable results

## 12. Limitations
- Dataset contains synthetic examples; real-world distribution shift is still possible.
- Context-sensitive phrases (for example, “dynamic” or “strong”) can produce false positives.
- Rewrite quality depends on external LLM output when API keys are provided.
- API security is currently a shared static key model (suitable for controlled environments, not full enterprise auth).

## 13. Roadmap
- Validation-set temperature fitting utility for confidence calibration (automated `T` selection)
- Calibration diagnostics (reliability plots / ECE) in evaluation workflow
- Stronger auth options (per-tenant keys or OAuth gateway integration)
- Expanded bias categories and domain adaptation datasets
- Observability for production (latency, error, and drift monitoring)

## 14. Tech Stack
- **Language:** Python
- **ML/NLP:** PyTorch, Hugging Face Transformers, Datasets, SeqEval, scikit-learn
- **API:** FastAPI, Uvicorn, Pydantic
- **UI:** Streamlit
- **LLM Rewrite Provider:** Anthropic SDK (with deterministic fallback rules)
- **Tooling:** pytest, Docker, Docker Compose

## 15. License
MIT License. See [`LICENSE`](LICENSE).
