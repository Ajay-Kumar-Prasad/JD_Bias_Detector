"""
dependencies.py — FastAPI dependency injection
Provides singleton instances of shared models/services
with lazy initialization on first use.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    try:
        from api.models.classifier import BiasClassifier
        from api.models.rewriter import BiasRewriter
        from api.models.scorer import BiasScorer
    except ModuleNotFoundError:
        # Support running from inside `api/` via `uvicorn main:app`.
        from models.classifier import BiasClassifier
        from models.rewriter import BiasRewriter
        from models.scorer import BiasScorer


classifier_model: "BiasClassifier | None" = None
rewriter_model: "BiasRewriter | None" = None
scorer_model: "BiasScorer | None" = None


def get_classifier() -> "BiasClassifier":
    global classifier_model
    if classifier_model is None:
        try:
            from api.models.classifier import BiasClassifier
        except ModuleNotFoundError:
            from models.classifier import BiasClassifier
        print("Loading model...")
        classifier_model = BiasClassifier()
    return classifier_model


def get_rewriter() -> "BiasRewriter":
    global rewriter_model
    if rewriter_model is None:
        try:
            from api.models.rewriter import BiasRewriter
        except ModuleNotFoundError:
            from models.rewriter import BiasRewriter
        rewriter_model = BiasRewriter()
    return rewriter_model


def get_scorer() -> "BiasScorer":
    global scorer_model
    if scorer_model is None:
        try:
            from api.models.scorer import BiasScorer
        except ModuleNotFoundError:
            from models.scorer import BiasScorer
        scorer_model = BiasScorer()
    return scorer_model
