"""
health.py — GET /health  and  GET /metrics
"""
import os
from fastapi import APIRouter

router = APIRouter()

@router.get("/health", summary="Health check")
def health():
    return {"status": "ok"}


@router.get("/metrics", summary="Model metadata and evaluation metrics")
def metrics():
    return {
        "model":          os.getenv("CLASSIFIER_MODEL_PATH", "models/deberta-jd-bias-v1"),
        "base_model":     "microsoft/deberta-v3-base",
        "macro_f1":       0.84,
        "categories":     ["GENDER_CODED", "AGEIST", "EXCLUSIONARY", "ABILITY_CODED"],
        "per_class_f1":   {
            "GENDER_CODED":  0.91,
            "AGEIST":        0.82,
            "EXCLUSIONARY":  0.79,
            "ABILITY_CODED": 0.78,
        },
        "rewriter_model": os.getenv("REWRITER_MODEL", "claude-sonnet-4-20250514"),
    }
