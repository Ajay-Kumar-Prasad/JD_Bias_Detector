"""
dependencies.py — FastAPI dependency injection
Provides singleton instances of the classifier and rewriter
so models are loaded once at startup, not per-request.
"""
from functools import lru_cache

try:
    from api.models.classifier import BiasClassifier
    from api.models.rewriter import BiasRewriter
    from api.models.scorer import BiasScorer
except ModuleNotFoundError:
    # Support running from inside `api/` via `uvicorn main:app`.
    from models.classifier import BiasClassifier
    from models.rewriter import BiasRewriter
    from models.scorer import BiasScorer


@lru_cache(maxsize=1)
def get_classifier() -> BiasClassifier:
    return BiasClassifier()


@lru_cache(maxsize=1)
def get_rewriter() -> BiasRewriter:
    return BiasRewriter()


@lru_cache(maxsize=1)
def get_scorer() -> BiasScorer:
    return BiasScorer()
