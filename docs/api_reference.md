# API Reference

Base URL: `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`

---

## Authentication

`/analyze/*` endpoints require API key header:

```http
x-api-key: your-secret-key
```

Configure with env var `API_KEY`.

---

## `GET /health`

Health check. Returns 200 if the API is running.

**Response**
```json
{ "status": "ok" }
```

---

## `GET /metrics`

Model metadata and evaluation metrics.

**Response**
```json
{
  "model":          "models/deberta-jd-bias-v2-clean",
  "base_model":     "microsoft/deberta-v3-base",
  "macro_f1":       0.9612,
  "categories":     ["GENDER_CODED", "AGEIST", "EXCLUSIONARY", "ABILITY_CODED"],
  "per_class_f1":   {
    "GENDER_CODED":  0.9114,
    "AGEIST":        1.0000,
    "EXCLUSIONARY":  0.9901,
    "ABILITY_CODED": 0.9677
  },
  "rewriter_model": "claude-sonnet-4-20250514"
}
```

---

## `POST /analyze/`

Analyzes a job description for biased language.

**Request body**
```json
{
  "text": "We are looking for a rockstar engineer who is young and hungry to crush it."
}
```

| Field | Type   | Required | Notes |
|-------|--------|----------|-------|
| text  | string | Yes      | Min 10 characters. Raw JD text. |

Required header:
```http
x-api-key: your-secret-key
```

**Response**
```json
{
  "inclusivity_score": 42,
  "flagged_spans": [
    {
      "text":        "rockstar",
      "start":       27,
      "end":         35,
      "category":    "EXCLUSIONARY",
      "confidence":  0.96,
      "explanation": "Vague hyperbolic language that signals cultural gatekeeping and discourages applicants who do not self-identify with the term.",
      "rewrite":     "skilled engineer"
    }
  ],
  "rewritten_text": "We are looking for a skilled engineer who is motivated and driven to excel.",
  "category_breakdown": {
    "GENDER_CODED":  1,
    "AGEIST":        1,
    "EXCLUSIONARY":  1,
    "ABILITY_CODED": 0
  }
}
```

### Response fields

| Field | Type | Description |
|---|---|---|
| `inclusivity_score` | int (0–100) | Overall inclusivity score. Higher is better. |
| `flagged_spans` | array | Each biased phrase found in the JD. |
| `flagged_spans[].text` | string | The exact biased phrase. |
| `flagged_spans[].start` | int | Char offset (start) in original text. |
| `flagged_spans[].end` | int | Char offset (end) in original text. |
| `flagged_spans[].category` | string | One of `GENDER_CODED`, `AGEIST`, `EXCLUSIONARY`, `ABILITY_CODED`. |
| `flagged_spans[].confidence` | float | Model confidence (0–1). |
| `flagged_spans[].explanation` | string | Why the phrase is biased. |
| `flagged_spans[].rewrite` | string | Neutral replacement suggestion. |
| `rewritten_text` | string | Full JD with all rewrites applied. |
| `category_breakdown` | object | Count of flags per category. |

### Error responses

| Status | Meaning |
|---|---|
| 401 | Missing `x-api-key` header. |
| 403 | Invalid API key. |
| 422 | Validation error — text too short or missing. |
| 500 | Internal server error — model or LLM failure. |

---

## `POST /analyze/batch`

Analyze multiple job descriptions in one request.

Required header:
```http
x-api-key: your-secret-key
```

**Request body**
```json
{
  "texts": [
    "We are looking for a rockstar engineer.",
    "We value collaboration and clear communication."
  ]
}
```

**Response**
```json
{
  "total": 2,
  "results": [
    {
      "inclusivity_score": 55,
      "flagged_spans": [],
      "rewritten_text": "We are looking for a skilled engineer.",
      "category_breakdown": {
        "GENDER_CODED": 0,
        "AGEIST": 0,
        "EXCLUSIONARY": 1,
        "ABILITY_CODED": 0
      }
    }
  ]
}
```

---

## Scoring formula

```
penalty = Σ (category_weight × confidence × length_normalizer)

length_normalizer = log(word_count) / log(200)

score = clamp(round(100 - penalty), 0, 100)
```

Category weights: `AGEIST=18`, `ABILITY_CODED=15`, `GENDER_CODED=12`, `EXCLUSIONARY=10`.

Longer JDs are normalised to avoid penalising them more than short ones for the same density of bias.
