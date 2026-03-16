"""
dependencies.py — FastAPI dependency injection
Provides singleton instances of the classifier and rewriter
so models are loaded once at startup, not per-request.
"""
from functools import lru_cache
from api.models.classifier import BiasClassifier
from api.models.rewriter   import BiasRewriter
from api.models.scorer     import BiasScorer


@lru_cache(maxsize=1)
def get_classifier() -> BiasClassifier:
    return BiasClassifier()


@lru_cache(maxsize=1)
def get_rewriter() -> BiasRewriter:
    return BiasRewriter()


@lru_cache(maxsize=1)
def get_scorer() -> BiasScorer:
    return BiasScorer()
