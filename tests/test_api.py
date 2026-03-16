"""
test_api.py — Integration tests for the FastAPI endpoints.
ML models are mocked to avoid loading weights in CI.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock

ENRICHED_SPANS = [
    {
        "text": "rockstar", "start": 22, "end": 30,
        "category": "EXCLUSIONARY", "confidence": 0.96,
        "rewrite": "skilled engineer",
        "explanation": "Vague hyperbolic language that signals cultural gatekeeping.",
    }
]


@pytest.fixture(scope="module")
def client():
    with patch("api.dependencies.BiasClassifier") as MockClf, \
         patch("api.dependencies.BiasRewriter")   as MockRw,  \
         patch("api.dependencies.BiasScorer")     as MockSc:

        MockClf.return_value.predict      = MagicMock(return_value=[ENRICHED_SPANS[0]])
        MockRw.return_value.rewrite_all   = AsyncMock(return_value=ENRICHED_SPANS)
        MockSc.return_value.score         = MagicMock(return_value=(55, {
            "GENDER_CODED": 0, "AGEIST": 0,
            "EXCLUSIONARY": 1, "ABILITY_CODED": 0,
        }))

        from api.main import app
        with TestClient(app) as c:
            yield c


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_metrics_endpoint(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "macro_f1"   in data
    assert "categories" in data


def test_analyze_returns_200(client):
    resp = client.post("/analyze/", json={
        "text": "We are looking for a rockstar who is young and hungry."
    })
    assert resp.status_code == 200


def test_analyze_response_structure(client):
    resp = client.post("/analyze/", json={
        "text": "We are looking for a rockstar who is young and hungry."
    })
    data = resp.json()
    assert "inclusivity_score"  in data
    assert "flagged_spans"      in data
    assert "rewritten_text"     in data
    assert "category_breakdown" in data


def test_analyze_span_fields(client):
    resp = client.post("/analyze/", json={
        "text": "We are looking for a rockstar who is young and hungry."
    })
    for span in resp.json()["flagged_spans"]:
        assert "text"        in span
        assert "category"    in span
        assert "confidence"  in span
        assert "rewrite"     in span
        assert "explanation" in span


def test_analyze_score_range(client):
    resp = client.post("/analyze/", json={
        "text": "We are looking for a rockstar who is young and hungry."
    })
    score = resp.json()["inclusivity_score"]
    assert 0 <= score <= 100


def test_analyze_rejects_short_text(client):
    resp = client.post("/analyze/", json={"text": "Hi"})
    assert resp.status_code == 422


def test_analyze_empty_text_rejected(client):
    resp = client.post("/analyze/", json={"text": ""})
    assert resp.status_code == 422
