"""
conftest.py — Shared pytest fixtures
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock

from api.main import app


@pytest.fixture(scope="session")
def client():
    """FastAPI test client with mocked ML dependencies."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_classifier():
    clf = MagicMock()
    clf.predict.return_value = [
        {
            "text": "rockstar", "start": 27, "end": 35,
            "category": "EXCLUSIONARY", "confidence": 0.96,
        }
    ]
    return clf


@pytest.fixture
def mock_rewriter():
    rw = MagicMock()
    rw.rewrite_all = AsyncMock(return_value=[
        {
            "text": "rockstar", "start": 27, "end": 35,
            "category": "EXCLUSIONARY", "confidence": 0.96,
            "rewrite": "skilled engineer",
            "explanation": "Vague hyperbolic language that signals cultural gatekeeping.",
        }
    ])
    return rw


@pytest.fixture
def sample_jd():
    return (
        "We are looking for a rockstar engineer who is young and hungry "
        "to crush it in a fast-paced environment."
    )
