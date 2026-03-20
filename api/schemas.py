"""
schemas.py — Pydantic request/response models for the API.
"""
from pydantic import BaseModel, Field, field_validator
from typing import List


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Raw job description text.")

    model_config = {"json_schema_extra": {
        "example": {
            "text": (
                "We are looking for a rockstar engineer who is young and hungry "
                "to crush it in a fast-paced environment."
            )
        }
    }}


class FlaggedSpan(BaseModel):
    text:        str
    start:       int
    end:         int
    category:    str   # GENDER_CODED | AGEIST | EXCLUSIONARY | ABILITY_CODED
    confidence:  float
    explanation: str
    rewrite:     str


class CategoryBreakdown(BaseModel):
    GENDER_CODED:  int = 0
    AGEIST:        int = 0
    EXCLUSIONARY:  int = 0
    ABILITY_CODED: int = 0


class AnalyzeResponse(BaseModel):
    inclusivity_score:  int
    flagged_spans:      List[FlaggedSpan]
    rewritten_text:     str
    category_breakdown: CategoryBreakdown

    model_config = {"json_schema_extra": {
        "example": {
            "inclusivity_score": 42,
            "flagged_spans": [{
                "text": "rockstar", "start": 27, "end": 35,
                "category": "EXCLUSIONARY", "confidence": 0.96,
                "explanation": "Vague hyperbolic language that signals cultural gatekeeping.",
                "rewrite": "skilled engineer",
            }],
            "rewritten_text": "We are looking for a skilled engineer...",
            "category_breakdown": {
                "GENDER_CODED": 1, "AGEIST": 1, "EXCLUSIONARY": 1, "ABILITY_CODED": 0
            }
        }
    }}


class BatchAnalyzeRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=200)

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, value: List[str]) -> List[str]:
        cleaned = [v.strip() for v in value if isinstance(v, str) and v.strip()]
        if not cleaned:
            raise ValueError("At least one non-empty text is required.")
        too_short = [t for t in cleaned if len(t) < 10]
        if too_short:
            raise ValueError("Each text must be at least 10 characters long.")
        return cleaned


class BatchAnalyzeResponse(BaseModel):
    total: int
    results: List[AnalyzeResponse]
