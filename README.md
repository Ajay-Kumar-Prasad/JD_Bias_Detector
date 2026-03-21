# 🚀 JD Bias Detector

A production-oriented NLP system that detects biased language in job descriptions, explains why it is problematic, and generates inclusive rewrites using a confidence-aware policy.

---

## 🌐 Live Demo

* 🔗 **Frontend (Streamlit):** https://jd-bias-detector.streamlit.app
* 🔗 **Backend API:** https://jd-bias-detector.onrender.com
* 📘 **API Docs (Swagger):** https://jd-bias-detector.onrender.com/docs
* 🤗 **Model (Hugging Face):** https://huggingface.co/Ajay-Kumar-Prasad/jd-bias-detector

---

## 🧠 Problem Statement

Manual review of job descriptions for bias is:

* slow
* inconsistent
* not scalable

Biased language reduces applications from qualified candidates and impacts diversity outcomes.

---

## 💡 Solution

JD Bias Detector provides an end-to-end system that:

* Detects biased spans at token level
* Classifies bias into defined categories
* Generates explanations and rewrites
* Applies confidence-aware rewrite policies
* Computes an inclusivity score
* Exposes functionality via API + UI

---

## ✨ Key Features

* 🔍 Token-level bias detection using **DeBERTa-v3**
* 🧩 Categories:

  * `GENDER_CODED`
  * `AGEIST`
  * `EXCLUSIONARY`
  * `ABILITY_CODED`
* 🎯 Confidence-aware rewrite modes:

  * `auto_replace`
  * `suggest`
  * `ignore`
* 📊 Inclusivity score (0–100)
* 📦 Batch CSV analysis (Streamlit)
* 🔐 API key protection
* 🐳 Dockerized deployment

---

## 🏗️ System Architecture

```text
Client (Recruiter / Analyst)
        |
        v
Streamlit UI
        |
        v
FastAPI Backend
  ├─ Classifier (DeBERTa)
  ├─ Rewriter (LLM + fallback)
  └─ Scorer
        |
        v
JSON Response
```

---

## ⚙️ ML / NLP Pipeline

1. Input validation
2. Token classification
3. Span merging & filtering
4. Rewrite + explanation generation
5. Confidence-based rewrite policy
6. Auto replacement
7. Inclusivity scoring

---

## 🤖 Model Details

* **Base Model:** `microsoft/deberta-v3-base`
* **Task:** Token Classification (IOB2)
* **Hosted on:** Hugging Face

### Labels

* GENDER_CODED
* AGEIST
* EXCLUSIONARY
* ABILITY_CODED
* O

### Training Highlights

* Weighted cross-entropy
* Early stopping
* LR scheduling + warmup
* Gradient accumulation

---

## 📊 Evaluation

**Test Set (n=315)**

| Metric    | Score      |
| --------- | ---------- |
| Macro F1  | **0.9612** |
| Precision | 0.9409     |
| Recall    | 0.9824     |

### Per-category F1

| Category      | F1     |
| ------------- | ------ |
| GENDER_CODED  | 0.9114 |
| AGEIST        | 1.0000 |
| EXCLUSIONARY  | 0.9901 |
| ABILITY_CODED | 0.9677 |

---

## 📁 Project Structure

```text
api/                FastAPI backend
app/                Streamlit frontend
training/           ML training pipeline
data/               datasets
docs/               evaluation + API docs
tests/              unit & integration tests
```

---

## ⚡ Quickstart (Local)

### Setup

```bash
git clone https://github.com/Ajay-Kumar-Prasad/JD_Bias_Detector.git
cd JD_Bias_Detector

python -m venv .venv
source .venv/bin/activate

pip install -r requirements-backend.txt
```

---

### Run API

```bash
uvicorn api.main:app --reload --port 8001
```

---

### Run UI

```bash
streamlit run app/streamlit_app.py
```

---

## 🧪 API Usage

```bash
curl -X POST https://jd-bias-detector.onrender.com/analyze/ \
-H "Content-Type: application/json" \
-H "x-api-key: your-secret-key" \
-d '{
  "text": "We are looking for a young rockstar engineer",
  "auto_rewrite_threshold": 0.85,
  "suggestion_threshold": 0.70
}'
```

---

## 🖥️ Demo Features

* Highlighted biased spans
* Rewrite suggestions
* Confidence tuning sliders
* Side-by-side comparison
* Batch CSV processing

---

## ⚠️ Limitations

* Synthetic data bias risk
* Context-sensitive false positives
* LLM rewrite variability
* Basic API key security

---

## 🛣️ Roadmap

* Confidence calibration (ECE, reliability plots)
* Improved authentication (OAuth / multi-tenant)
* More bias categories
* Production monitoring (latency, drift)

---

## 🧰 Tech Stack

* **ML:** PyTorch, Transformers
* **Backend:** FastAPI, Uvicorn
* **Frontend:** Streamlit
* **LLM:** Anthropic API
* **Infra:** Render, Streamlit Cloud
* **Testing:** pytest

---

## 📄 License

MIT License
