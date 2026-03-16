"""
test_rewriter.py — Unit tests for the rewriter module.
LLM calls are mocked so no API key is needed in CI.
"""
import pytest
import asyncio
from unittest.mock import patch, MagicMock
from api.models.rewriter import BiasRewriter


SAMPLE_SPAN = {
    "text": "rockstar", "start": 10, "end": 18,
    "category": "EXCLUSIONARY", "confidence": 0.96,
}

SAMPLE_TEXT = "We are looking for a rockstar engineer."


@pytest.fixture
def rewriter_with_mock_llm():
    rw = BiasRewriter.__new__(BiasRewriter)
    rw._client = MagicMock()
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text='{"rewrite": "skilled engineer", "explanation": "Vague jargon."}')]
    rw._client.messages.create.return_value = mock_msg
    return rw


@pytest.fixture
def rewriter_no_key():
    rw = BiasRewriter.__new__(BiasRewriter)
    rw._client = None
    return rw


def test_rewrite_span_returns_enriched_dict(rewriter_with_mock_llm):
    result = rewriter_with_mock_llm.rewrite_span(SAMPLE_TEXT, SAMPLE_SPAN.copy())
    assert "rewrite"     in result
    assert "explanation" in result


def test_rewrite_span_preserves_original_fields(rewriter_with_mock_llm):
    result = rewriter_with_mock_llm.rewrite_span(SAMPLE_TEXT, SAMPLE_SPAN.copy())
    assert result["text"]     == SAMPLE_SPAN["text"]
    assert result["category"] == SAMPLE_SPAN["category"]
    assert result["start"]    == SAMPLE_SPAN["start"]


def test_fallback_used_when_no_client(rewriter_no_key):
    result = rewriter_no_key.rewrite_span(SAMPLE_TEXT, SAMPLE_SPAN.copy())
    assert "rewrite"     in result
    assert "explanation" in result
    assert len(result["rewrite"]) > 0


def test_fallback_on_llm_error(rewriter_with_mock_llm):
    rewriter_with_mock_llm._client.messages.create.side_effect = Exception("API down")
    result = rewriter_with_mock_llm.rewrite_span(SAMPLE_TEXT, SAMPLE_SPAN.copy())
    assert "rewrite" in result      # fallback kicks in


def test_rewrite_all_returns_same_count(rewriter_with_mock_llm):
    spans = [SAMPLE_SPAN.copy(), {**SAMPLE_SPAN, "text": "young", "category": "AGEIST"}]
    result = asyncio.run(rewriter_with_mock_llm.rewrite_all(SAMPLE_TEXT, spans))
    assert len(result) == 2
