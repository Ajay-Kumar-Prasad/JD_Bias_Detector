"""
test_scorer.py — Unit tests for the scoring module.
"""
import pytest
from api.models.scorer import BiasScorer

scorer = BiasScorer()

CLEAN_TEXT = "We are seeking a skilled engineer to join our collaborative team."

BIASED_SPANS = [
    {"text": "rockstar",         "category": "EXCLUSIONARY",  "confidence": 0.96},
    {"text": "young and hungry", "category": "AGEIST",        "confidence": 0.91},
    {"text": "aggressive",       "category": "GENDER_CODED",  "confidence": 0.88},
]


def test_score_returns_tuple():
    score, breakdown = scorer.score(CLEAN_TEXT, [])
    assert isinstance(score, int)
    assert isinstance(breakdown, dict)


def test_clean_jd_scores_100():
    score, _ = scorer.score(CLEAN_TEXT, [])
    assert score == 100


def test_biased_jd_scores_lower():
    score, _ = scorer.score(CLEAN_TEXT, BIASED_SPANS)
    assert score < 100


def test_score_clamped_between_0_and_100():
    many_spans = BIASED_SPANS * 20    # extreme penalty
    score, _ = scorer.score(CLEAN_TEXT, many_spans)
    assert 0 <= score <= 100


def test_breakdown_counts_correctly():
    _, breakdown = scorer.score(CLEAN_TEXT, BIASED_SPANS)
    assert breakdown["EXCLUSIONARY"] == 1
    assert breakdown["AGEIST"]       == 1
    assert breakdown["GENDER_CODED"] == 1
    assert breakdown["ABILITY_CODED"]== 0


def test_breakdown_has_all_categories():
    _, breakdown = scorer.score(CLEAN_TEXT, [])
    assert set(breakdown.keys()) == {"GENDER_CODED", "AGEIST", "EXCLUSIONARY", "ABILITY_CODED"}
