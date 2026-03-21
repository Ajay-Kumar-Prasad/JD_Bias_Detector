"""
analyze.py — POST /analyze
Full pipeline: classify → rewrite → score → return.
"""
import os
import re
from typing import TYPE_CHECKING
from fastapi import APIRouter, Depends

try:
    from api.schemas import (
        AnalyzeRequest,
        AnalyzeResponse,
        BatchAnalyzeRequest,
        BatchAnalyzeResponse,
        CategoryBreakdown,
    )
    from api.dependencies import get_classifier, get_rewriter, get_scorer
except ModuleNotFoundError:
    # Support running from inside `api/` via `uvicorn main:app`.
    from schemas import (
        AnalyzeRequest,
        AnalyzeResponse,
        BatchAnalyzeRequest,
        BatchAnalyzeResponse,
        CategoryBreakdown,
    )
    from dependencies import get_classifier, get_rewriter, get_scorer

if TYPE_CHECKING:
    try:
        from api.models.classifier import BiasClassifier
        from api.models.rewriter import BiasRewriter
        from api.models.scorer import BiasScorer
    except ModuleNotFoundError:
        from models.classifier import BiasClassifier
        from models.rewriter import BiasRewriter
        from models.scorer import BiasScorer

router = APIRouter()
MAX_SPANS = 10
AUTO_REWRITE_MIN_CONFIDENCE = float(os.getenv("AUTO_REWRITE_MIN_CONFIDENCE", "0.85"))
SUGGESTION_MIN_CONFIDENCE = float(os.getenv("SUGGESTION_MIN_CONFIDENCE", "0.70"))


def fix_articles(text: str) -> str:
    return re.sub(r"\ba ([aeiouAEIOU])", r"an \1", text)


def deduplicate_phrases(text: str) -> str:
    # Compress repeated conjunction patterns: "driven and driven" -> "driven"
    return re.sub(r"\b(\w+)( and \1)+\b", r"\1", text, flags=re.IGNORECASE)


def fix_redundant_prepositions(text: str) -> str:
    return text.replace("in the role in a", "in a")


def _cleanup_text(text: str) -> str:
    text = fix_redundant_prepositions(text)
    text = fix_articles(text)
    text = deduplicate_phrases(text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([(\[{])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]}])", r"\1", text)
    return text.strip()


def _build_rewritten_text(original: str, spans: list[dict]) -> str:
    """
    Applies rewrites to the original text in reverse order
    so char offsets remain valid after each replacement.
    """
    text = original
    for span in sorted(spans, key=lambda s: s["start"], reverse=True):
        rewrite = str(span.get("rewrite", span["text"])).strip()
        start = int(span["start"])
        end = int(span["end"])

        left = text[:start]
        right = text[end:]

        # Keep token boundaries readable when replacement lengths differ.
        if left and left[-1].isalnum() and rewrite and rewrite[0].isalnum():
            rewrite = " " + rewrite
        if right and right[:1].isalnum() and rewrite and rewrite[-1].isalnum():
            rewrite = rewrite + " "

        text = left + rewrite + right
    return _cleanup_text(text)


def _resolve_policy_thresholds(
    auto_threshold: float | None,
    suggestion_threshold: float | None,
) -> tuple[float, float]:
    auto = AUTO_REWRITE_MIN_CONFIDENCE if auto_threshold is None else float(auto_threshold)
    suggest = SUGGESTION_MIN_CONFIDENCE if suggestion_threshold is None else float(suggestion_threshold)
    auto = min(max(auto, 0.0), 1.0)
    suggest = min(max(suggest, 0.0), 1.0)
    if suggest > auto:
        suggest = auto
    return auto, suggest


def _apply_rewrite_policy(
    spans: list[dict],
    auto_threshold: float,
    suggestion_threshold: float,
) -> list[dict]:
    """
    Assign rewrite behavior based on confidence.
    - high confidence: auto replace
    - medium confidence: suggestion only
    - low confidence: ignore
    """
    policy_spans: list[dict] = []
    for span in spans:
        enriched = span.copy()
        rewrite_confidence = float(enriched.get("rewrite_confidence", enriched["confidence"]))
        enriched["rewrite_confidence"] = round(rewrite_confidence, 4)

        if rewrite_confidence >= auto_threshold:
            enriched["rewrite_mode"] = "auto_replace"
        elif rewrite_confidence >= suggestion_threshold:
            enriched["rewrite_mode"] = "suggest"
        else:
            enriched["rewrite_mode"] = "ignore"

        policy_spans.append(enriched)
    return policy_spans


async def _analyze_text(
    text: str,
    classifier: "BiasClassifier",
    rewriter: "BiasRewriter",
    scorer: "BiasScorer",
    auto_threshold: float | None = None,
    suggestion_threshold: float | None = None,
) -> AnalyzeResponse:
    auto_threshold, suggestion_threshold = _resolve_policy_thresholds(
        auto_threshold, suggestion_threshold
    )
    spans = classifier.predict(text)
    spans = sorted(spans, key=lambda x: x["confidence"], reverse=True)
    spans = spans[:MAX_SPANS]
    enriched_spans = await rewriter.rewrite_all(text, spans)
    policy_spans = _apply_rewrite_policy(
        enriched_spans, auto_threshold=auto_threshold, suggestion_threshold=suggestion_threshold
    )
    scoreable_spans = [s for s in policy_spans if s.get("rewrite_mode") != "ignore"]
    score, breakdown = scorer.score(text, scoreable_spans)
    auto_replace_spans = [s for s in policy_spans if s.get("rewrite_mode") == "auto_replace"]
    rewritten_text = _build_rewritten_text(text, auto_replace_spans)
    return AnalyzeResponse(
        inclusivity_score=score,
        flagged_spans=policy_spans,
        rewritten_text=rewritten_text,
        category_breakdown=CategoryBreakdown(**breakdown),
    )


@router.post("/", response_model=AnalyzeResponse, summary="Analyze a job description for bias")
async def analyze(
    request:    AnalyzeRequest,
    classifier: "BiasClassifier" = Depends(get_classifier),
    rewriter:   "BiasRewriter"   = Depends(get_rewriter),
    scorer:     "BiasScorer"     = Depends(get_scorer),
):
    return await _analyze_text(
        request.text,
        classifier,
        rewriter,
        scorer,
        auto_threshold=request.auto_rewrite_threshold,
        suggestion_threshold=request.suggestion_threshold,
    )


@router.post(
    "/batch",
    response_model=BatchAnalyzeResponse,
    summary="Analyze multiple job descriptions for bias",
)
async def analyze_batch(
    request: BatchAnalyzeRequest,
    classifier: "BiasClassifier" = Depends(get_classifier),
    rewriter: "BiasRewriter" = Depends(get_rewriter),
    scorer: "BiasScorer" = Depends(get_scorer),
):
    results = []
    for text in request.texts:
        results.append(
            await _analyze_text(
                text,
                classifier,
                rewriter,
                scorer,
                auto_threshold=request.auto_rewrite_threshold,
                suggestion_threshold=request.suggestion_threshold,
            )
        )
    return BatchAnalyzeResponse(total=len(results), results=results)
