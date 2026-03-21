"""
dependencies.py — FastAPI dependency injection
Provides singleton instances of shared models/services
with lazy initialization on first use.
"""
from functools import lru_cache
from typing import Optional

try:
    from api.models.classifier import BiasClassifier
    from api.models.rewriter import BiasRewriter
    from api.models.scorer import BiasScorer
except ModuleNotFoundError:
    # Support running from inside `api/` via `uvicorn main:app`.
    from models.classifier import BiasClassifier
    from models.rewriter import BiasRewriter
    from models.scorer import BiasScorer


model: Optional[BiasClassifier] = None


def get_classifier() -> BiasClassifier:
    global model
    if model is None:
        print("Loading model...")
        model = BiasClassifier()
    return model


@lru_cache(maxsize=1)
def get_rewriter() -> BiasRewriter:
    return BiasRewriter()


@lru_cache(maxsize=1)
def get_scorer() -> BiasScorer:
    return BiasScorer()
