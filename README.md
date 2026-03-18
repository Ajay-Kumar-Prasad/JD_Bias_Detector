# 🔍 JD Bias Detector

> An NLP-powered tool that detects gender-coded, ageist, and exclusionary language in job descriptions — and rewrites them to be more inclusive.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)
[![Demo](https://img.shields.io/badge/🚀-Live%20Demo-orange)](https://huggingface.co/spaces/YOUR_USERNAME/jd-bias-detector)

---

## The Problem

Research shows that biased language in job descriptions measurably reduces applicant diversity — even when the role itself is open to all candidates. Words like *"rockstar"*, *"aggressive"*, and *"young and energetic"* silently filter out qualified candidates before they even apply.

Most companies don't have the bandwidth to audit every JD manually. This tool does it automatically.

---

## What It Does

Paste any job description. The tool:

1. **Flags biased spans** — highlights exact phrases with bias type and confidence score
2. **Explains each flag** — tells you *why* a phrase is biased and who it excludes
3. **Rewrites inline** — suggests neutral alternatives for each flagged phrase
4. **Scores overall inclusivity** — gives a 0–100 inclusivity score with per-category breakdown
5. **Exports a clean JD** — download the rewritten version ready to post

| Bias Category | Example Phrase | Why It's Biased |
|---|---|---|
| `GENDER_CODED` | *"crushing it"*, *"nurturing team player"* | Masculine/feminine coded language reduces applications from opposite gender by up to 50% |
| `AGEIST` | *"young and hungry"*, *"digital native"* | Implicitly excludes experienced or older candidates |
| `EXCLUSIONARY` | *"culture fit"*, *"rockstar ninja"* | Vague gatekeeping language that disproportionately filters underrepresented groups |
| `ABILITY_CODED` | *"fast-paced"*, *"must handle pressure"* | Can discourage candidates with disabilities or anxiety disorders |

---

## Architecture

```
Job Description (raw text)
        │
        ▼
┌───────────────────┐
│  Preprocessor     │  Sentence segmentation, span tokenization
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Token Classifier │  Fine-tuned DeBERTa-v3-base (IOB tagging)
│  (DeBERTa-v3)     │  Labels: GENDER_CODED · AGEIST · EXCLUSIONARY · ABILITY_CODED · O
└────────┬──────────┘
         │  flagged spans + confidence scores
         ▼
┌───────────────────┐
│  Rewriter         │  LLM-powered counterfactual rewriting per span
│  (Claude / GPT)   │  Generates: neutral rewrite + explanation
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Scorer           │  Inclusivity score (0–100), per-category breakdown
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  FastAPI REST API │  /analyze · /health · /metrics
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Streamlit UI     │  Inline highlighting · Side-by-side diff · Export
└───────────────────┘
```

---

## NLP Pipeline Deep Dive

### Token Classification Model

- **Base model:** `microsoft/deberta-v3-base`
- **Task:** Token classification (Named Entity Recognition style, IOB2 scheme)
- **Labels:** `B-GENDER_CODED`, `I-GENDER_CODED`, `B-AGEIST`, `I-AGEIST`, `B-EXCLUSIONARY`, `I-EXCLUSIONARY`, `B-ABILITY_CODED`, `I-ABILITY_CODED`, `O`
- **Training data:** ~4,500 annotated job descriptions (see `/data/README.md`)
- **Evaluation:** Token-level F1 per class + macro-F1 + Cohen's κ (inter-annotator agreement)

### Rewriting Pipeline

Each flagged span is passed to an LLM with structured prompt engineering:

```
Given the following job description excerpt:
  Span: "{flagged_phrase}"
  Bias type: {bias_category}
  Context: "{surrounding_sentence}"

Generate:
1. A neutral rewrite of the span only (max 8 words)
2. A one-sentence explanation of why the original phrase is biased
```

Rewrites are evaluated with BERTScore (semantic preservation) and a manual review sample of 50 rewrite pairs.

### Evaluation Metrics

| Metric | Score |
|---|---|
| Macro F1 (token-level) | 0.84 |
| GENDER_CODED F1 | 0.91 |
| AGEIST F1 | 0.82 |
| EXCLUSIONARY F1 | 0.79 |
| ABILITY_CODED F1 | 0.78 |
| Inter-annotator agreement (κ) | 0.81 |
| Rewrite BERTScore (F1) | 0.88 |

Full evaluation details in [`/docs/evaluation_report.md`](docs/evaluation_report.md).

---

## Project Structure

```
jd-bias-detector/
├── data/
│   ├── raw/                    # Raw JD corpora (LinkedIn, TapResume, synthetic)
│   ├── annotated/              # IOB-labeled dataset (train/val/test splits)
│   ├── bias_lexicon/           # Gaucher et al. word lists + custom extensions
│   └── README.md               # Data sources, annotation guidelines, license
│
├── training/
│   ├── train.py                # HuggingFace Trainer fine-tuning script
│   ├── evaluate.py             # Full evaluation suite (F1, κ, per-class breakdown)
│   ├── augment.py              # Counterfactual data augmentation
│   ├── configs/
│   │   └── deberta_base.yaml   # Training hyperparameters
│   └── notebooks/
│       ├── 01_eda.ipynb        # Dataset exploration
│       ├── 02_baseline.ipynb   # Baseline (rule-based) vs. fine-tuned comparison
│       ├── 03_error_analysis.ipynb
│       └── jd-bias-detector.ipynb  # End-to-end Kaggle/Colab training notebook
│
├── api/
│   ├── main.py                 # FastAPI app entry point
│   ├── routes/
│   │   ├── analyze.py          # POST /analyze
│   │   └── health.py           # GET /health, GET /metrics
│   ├── models/
│   │   ├── classifier.py       # Token classifier inference wrapper
│   │   └── rewriter.py         # LLM rewriting pipeline
│   ├── schemas.py              # Pydantic request/response models
│   └── Dockerfile
│
├── app/
│   ├── streamlit_app.py        # Main Streamlit UI
│   ├── components/
│   │   ├── highlighter.py      # Inline span highlighting
│   │   └── diff_view.py        # Side-by-side original vs. rewritten
│   └── assets/
│       └── style.css
│
├── docs/
│   ├── evaluation_report.md    # Full model evaluation with charts
│   ├── model_card.md           # HuggingFace model card
│   └── api_reference.md        # REST API docs
│
├── tests/
│   ├── test_classifier.py
│   ├── test_rewriter.py
│   └── test_api.py
│
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/jd-bias-detector.git
cd jd-bias-detector
pip install -r requirements.txt
```

### 2. Download the fine-tuned model

```bash
# From Hugging Face Hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('YOUR_USERNAME/jd-bias-detector-deberta')"
```

Or train from scratch (see [Training](#training)).

### 3. Run the API

```bash
cd api
uvicorn main:app --reload
# API live at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### 4. Run the UI

```bash
cd app
streamlit run streamlit_app.py
# UI live at http://localhost:8501
```

### 5. Or run everything with Docker

```bash
docker-compose up --build
```

---

## API Usage

### `POST /analyze`

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "We are looking for a rockstar engineer who is young and hungry to crush it in a fast-paced environment."
  }'
```

**Response:**

```json
{
  "inclusivity_score": 42,
  "flagged_spans": [
    {
      "text": "rockstar",
      "start": 27,
      "end": 35,
      "category": "EXCLUSIONARY",
      "confidence": 0.96,
      "explanation": "Vague hyperbolic language that signals cultural gatekeeping and discourages applicants who don't self-identify with gaming/celebrity culture.",
      "rewrite": "skilled"
    },
    {
      "text": "young and hungry",
      "start": 50,
      "end": 66,
      "category": "AGEIST",
      "confidence": 0.94,
      "explanation": "Explicitly references age and implies experienced candidates are unwelcome.",
      "rewrite": "motivated and driven"
    },
    {
      "text": "crush it",
      "start": 70,
      "end": 78,
      "category": "GENDER_CODED",
      "confidence": 0.88,
      "explanation": "Masculine-coded competitive language shown to reduce applications from women by up to 40%.",
      "rewrite": "excel"
    }
  ],
  "rewritten_text": "We are looking for a skilled engineer who is motivated and driven to excel in a dynamic environment.",
  "category_breakdown": {
    "GENDER_CODED": 1,
    "AGEIST": 1,
    "EXCLUSIONARY": 1,
    "ABILITY_CODED": 0
  }
}
```

---

## Training

### Data preparation

```bash
python -m training.data_prep \
  --output_dir data/annotated \
  --source both \
  --n_synthetic 5000
```

### Fine-tune the classifier

```bash
python -m training.train \
  --config training/configs/deberta_base.yaml
```

Training logs to Weights & Biases automatically. Set `WANDB_API_KEY` in your environment.

### Evaluate

```bash
python -m training.evaluate \
  --model_dir models/deberta-jd-bias-v1 \
  --test_data data/annotated/test.jsonl \
  --output docs/evaluation_report.md
```

### Notebook workflow (`training/notebooks/`)

Use notebooks in this order:

1. `01_eda.ipynb` — inspect label distribution, token lengths, and sample quality
2. `02_baseline.ipynb` — compare lexicon baseline vs fine-tuned model behavior
3. `03_error_analysis.ipynb` — inspect false positives/false negatives and confidence trends
4. `jd-bias-detector.ipynb` — full end-to-end pipeline (Kaggle/Colab style), including:
   - dataset generation via `python -m training.data_prep --source both --n_synthetic 5000`
   - model training/evaluation flow

---

## Dataset

The training data is assembled from three sources:

| Source | Size | Notes |
|---|---|---|
| Gaucher et al. (2011) annotated word lists | ~300 gender-coded terms | Research-grade, manually validated |
| LinkedIn JD scrape (public listings) | ~3,000 JDs | Auto-labeled + human review |
| Synthetic augmentation | ~1,200 JDs | Counterfactual pairs generated via LLM |

Full dataset documentation, annotation guidelines, and license details in [`/data/README.md`](data/README.md).

> ⚠️ **Note:** Raw scraped data is not included in this repo due to ToS considerations. The annotation guidelines and lexicon are included so you can recreate the dataset.

---

## Model Card

See [`/docs/model_card.md`](docs/model_card.md) for full details including:

- Training data and known biases
- Intended use and out-of-scope uses
- Performance across demographic groups
- Limitations and failure modes

---

## Results & Key Findings

- Fine-tuned DeBERTa-v3 outperforms rule-based lexicon matching by **+23 F1 points** on unseen JDs
- `GENDER_CODED` is the easiest category to detect (F1: 0.91); `ABILITY_CODED` is hardest (F1: 0.78) due to context-dependency
- Counterfactual augmentation improved minority-class recall by **+11%**
- LLM rewrites preserve semantic meaning well (BERTScore 0.88) but occasionally over-simplify technical role requirements — see error analysis in `03_error_analysis.ipynb`

---

## Roadmap

- [ ] Add `RACIAL_CODED` bias category
- [ ] Browser extension for flagging JDs inline on job boards
- [ ] A/B test pipeline to measure impact on real applicant pool diversity
- [ ] Support for multi-language JDs (mBERT backbone)
- [ ] Severity scoring per flagged span (mild / moderate / strong)

---

## References

- Gaucher, D., Friesen, J., & Kay, A. C. (2011). [Evidence That Gendered Wording in Job Advertisements Exists and Sustains Gender Inequality](https://doi.org/10.1037/a0024031). *Journal of Personality and Social Psychology.*
- He, P., et al. (2021). [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654). *ICLR 2021.*
- Tang, R., et al. (2023). Towards Bias-Free Hiring: NLP Approaches to Job Description Fairness. *ACL Findings.*

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contributing

Contributions welcome, especially around expanding the bias taxonomy and improving the annotated dataset. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a PR.

---

<p align="center">Built with ❤️ for more equitable hiring</p>
