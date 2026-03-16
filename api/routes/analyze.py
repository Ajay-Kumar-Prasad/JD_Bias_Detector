"""
analyze.py — POST /analyze
Full pipeline: classify → rewrite → score → return.
"""
from fastapi import APIRouter, Depends
from api.schemas import AnalyzeRequest, AnalyzeResponse, CategoryBreakdown
from api.models.classifier import BiasClassifier
from api.models.rewriter   import BiasRewriter
from api.models.scorer     import BiasScorer
from api.dependencies      import get_classifier, get_rewriter, get_scorer

router = APIRouter()


def _build_rewritten_text(original: str, spans: list[dict]) -> str:
    """
    Applies rewrites to the original text in reverse order
    so char offsets remain valid after each replacement.
    """
    text = original
    for span in sorted(spans, key=lambda s: s["start"], reverse=True):
        rewrite = span.get("rewrite", span["text"])
        text = text[: span["start"]] + rewrite + text[span["end"]:]
    return text


@router.post("/", response_model=AnalyzeResponse, summary="Analyze a job description for bias")
async def analyze(
    request:    AnalyzeRequest,
    classifier: BiasClassifier = Depends(get_classifier),
    rewriter:   BiasRewriter   = Depends(get_rewriter),
    scorer:     BiasScorer     = Depends(get_scorer),
):
    # 1. Detect biased spans
    spans = classifier.predict(request.text)

    # 2. Rewrite + explain each span (concurrent LLM calls)
    enriched_spans = await rewriter.rewrite_all(request.text, spans)

    # 3. Score
    score, breakdown = scorer.score(request.text, enriched_spans)

    # 4. Apply rewrites to produce clean JD
    rewritten_text = _build_rewritten_text(request.text, enriched_spans)

    return AnalyzeResponse(
        inclusivity_score=score,
        flagged_spans=enriched_spans,
        rewritten_text=rewritten_text,
        category_breakdown=CategoryBreakdown(**breakdown),
    )
