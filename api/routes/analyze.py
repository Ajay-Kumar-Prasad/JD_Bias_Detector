"""
analyze.py — POST /analyze
Full pipeline: classify → rewrite → score → return.
"""
import re
from fastapi import APIRouter, Depends
from api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchAnalyzeRequest,
    BatchAnalyzeResponse,
    CategoryBreakdown,
)
from api.models.classifier import BiasClassifier
from api.models.rewriter   import BiasRewriter
from api.models.scorer     import BiasScorer
from api.dependencies      import get_classifier, get_rewriter, get_scorer

router = APIRouter()


def fix_articles(text: str) -> str:
    return re.sub(r"\ba ([aeiouAEIOU])", r"an \1", text)


def deduplicate_phrases(text: str) -> str:
    # Compress repeated conjunction patterns: "driven and driven" -> "driven"
    return re.sub(r"\b(\w+)( and \1)+\b", r"\1", text, flags=re.IGNORECASE)


def _cleanup_text(text: str) -> str:
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


async def _analyze_text(
    text: str,
    classifier: BiasClassifier,
    rewriter: BiasRewriter,
    scorer: BiasScorer,
) -> AnalyzeResponse:
    spans = classifier.predict(text)
    enriched_spans = await rewriter.rewrite_all(text, spans)
    score, breakdown = scorer.score(text, enriched_spans)
    rewritten_text = _build_rewritten_text(text, enriched_spans)
    return AnalyzeResponse(
        inclusivity_score=score,
        flagged_spans=enriched_spans,
        rewritten_text=rewritten_text,
        category_breakdown=CategoryBreakdown(**breakdown),
    )


@router.post("/", response_model=AnalyzeResponse, summary="Analyze a job description for bias")
async def analyze(
    request:    AnalyzeRequest,
    classifier: BiasClassifier = Depends(get_classifier),
    rewriter:   BiasRewriter   = Depends(get_rewriter),
    scorer:     BiasScorer     = Depends(get_scorer),
):
    return await _analyze_text(request.text, classifier, rewriter, scorer)


@router.post(
    "/batch",
    response_model=BatchAnalyzeResponse,
    summary="Analyze multiple job descriptions for bias",
)
async def analyze_batch(
    request: BatchAnalyzeRequest,
    classifier: BiasClassifier = Depends(get_classifier),
    rewriter: BiasRewriter = Depends(get_rewriter),
    scorer: BiasScorer = Depends(get_scorer),
):
    results = []
    for text in request.texts:
        results.append(await _analyze_text(text, classifier, rewriter, scorer))
    return BatchAnalyzeResponse(total=len(results), results=results)
