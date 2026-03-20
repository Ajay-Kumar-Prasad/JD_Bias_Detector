"""
health.py — GET /health  and  GET /metrics
"""
import os
import re
from fastapi import APIRouter

router = APIRouter()

@router.get("/health", summary="Health check")
def health():
    return {"status": "ok"}


@router.get("/metrics", summary="Model metadata and evaluation metrics")
def metrics():
    report_path = os.getenv("EVALUATION_REPORT_PATH", "docs/evaluation_report.md")
    macro_f1 = None
    per_class_f1 = {}
    try:
        report = open(report_path, encoding="utf-8").read()
        macro_match = re.search(r"\| Macro F1 \| \*\*(\d+\.\d+)\*\* \|", report)
        if macro_match:
            macro_f1 = float(macro_match.group(1))

        for category in ["GENDER_CODED", "AGEIST", "EXCLUSIONARY", "ABILITY_CODED"]:
            row = re.search(
                rf"^\s*{category}\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+\d+",
                report,
                flags=re.MULTILINE,
            )
            if row:
                per_class_f1[category] = float(row.group(3))
    except Exception:
        pass

    return {
        "model":          os.getenv("CLASSIFIER_MODEL_PATH", "models/deberta-jd-bias-v2-clean"),
        "base_model":     "microsoft/deberta-v3-base",
        "macro_f1":       macro_f1,
        "categories":     ["GENDER_CODED", "AGEIST", "EXCLUSIONARY", "ABILITY_CODED"],
        "per_class_f1":   per_class_f1,
        "rewriter_model": os.getenv("REWRITER_MODEL", "claude-sonnet-4-20250514"),
    }
