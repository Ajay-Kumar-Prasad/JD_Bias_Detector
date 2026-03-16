"""
test_classifier.py — Unit tests for the token classifier wrapper.
Uses a lightweight spaCy/regex stub so no GPU is needed in CI.
"""
import pytest
from unittest.mock import patch, MagicMock
from api.models.classifier import BiasClassifier


MOCK_PIPELINE_OUTPUT = [
    {"word": "rockstar", "start": 10, "end": 18,
     "entity_group": "EXCLUSIONARY", "score": 0.96},
    {"word": "young and hungry", "start": 26, "end": 42,
     "entity_group": "AGEIST", "score": 0.91},
]


@pytest.fixture
def classifier():
    with patch("api.models.classifier.pipeline") as mock_pipe:
        mock_pipe.return_value = MagicMock(return_value=MOCK_PIPELINE_OUTPUT)
        clf = BiasClassifier()
        clf._pipe = MagicMock(return_value=MOCK_PIPELINE_OUTPUT)
    return clf


def test_predict_returns_list(classifier):
    result = classifier.predict("We need a rockstar who is young and hungry.")
    assert isinstance(result, list)


def test_predict_span_structure(classifier):
    result = classifier.predict("We need a rockstar who is young and hungry.")
    assert len(result) == 2
    span = result[0]
    assert "text"       in span
    assert "start"      in span
    assert "end"        in span
    assert "category"   in span
    assert "confidence" in span


def test_predict_confidence_range(classifier):
    result = classifier.predict("We need a rockstar.")
    for span in result:
        assert 0.0 <= span["confidence"] <= 1.0


def test_predict_valid_categories(classifier):
    VALID = {"GENDER_CODED", "AGEIST", "EXCLUSIONARY", "ABILITY_CODED"}
    result = classifier.predict("We need a rockstar who is young and hungry.")
    for span in result:
        assert span["category"] in VALID


def test_predict_empty_text(classifier):
    classifier._pipe.return_value = []
    result = classifier.predict("No bias here.")
    assert result == []
