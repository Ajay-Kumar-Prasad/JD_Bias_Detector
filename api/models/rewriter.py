"""
rewriter.py
Uses the Anthropic API to generate a neutral rewrite and explanation
for each flagged bias span. Falls back gracefully if the API is unavailable.
"""
import os
import json
import asyncio
import re
import anthropic

REWRITER_MODEL = os.getenv("REWRITER_MODEL", "claude-sonnet-4-20250514")

SYSTEM_PROMPT = (
    "You are a hiring language expert specialising in inclusive job description writing. "
    "You respond ONLY with a valid JSON object — no preamble, no markdown fences."
)

USER_TEMPLATE = """\
A job description contains a biased phrase. Provide:
1. A neutral drop-in rewrite (max 6 words, same grammatical role)
2. One clear sentence explaining why the original is biased and who it may exclude

Phrase: "{phrase}"
Bias type: {category}
Context (surrounding text): "{context}"

Respond with this exact JSON structure:
{{"rewrite": "...", "explanation": "..."}}"""

REWRITE_TEMPLATES = {
    "GENDER_CODED": {
        "crush it": "excel in the role",
        "rockstar": "highly skilled",
        "aggressive": "results-oriented",
    },
    "AGEIST": {
        "young and hungry": "motivated and driven",
        "young, hungry": "motivated and driven",
        "young professional": "early-career professional",
        "recent graduate": "entry-level candidate",
        "digital native": "comfortable with digital tools",
    },
    "ABILITY_CODED": {
        "fast-paced": "dynamic",
        "high pressure": "collaborative",
    },
    "EXCLUSIONARY": {
        "ninja": "specialist",
        "rockstar": "highly skilled",
    },
}

DEFAULT_EXPLANATIONS = {
    "GENDER_CODED": "This phrase uses gender-coded language that may discourage applicants.",
    "AGEIST": "This phrase implies age preference, which may exclude experienced candidates.",
    "EXCLUSIONARY": "This phrase uses exclusionary jargon that may deter underrepresented candidates.",
    "ABILITY_CODED": "This phrase may discourage candidates with disabilities or health conditions.",
}


class BiasRewriter:
    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("[Rewriter] WARNING: ANTHROPIC_API_KEY not set. Using fallback rewrites.")
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else None

    def _context_window(self, text: str, start: int, end: int, window: int = 80) -> str:
        """Returns text surrounding the span for context."""
        return text[max(0, start - window): end + window].strip()

    def _fallback(self, span: dict) -> dict:
        """Returns a rule-based fallback when the LLM is unavailable."""
        cat = span.get("category", "EXCLUSIONARY")
        rewrite = self._template_rewrite(span.get("text", ""), cat) or {
            "GENDER_CODED": "skilled professional",
            "AGEIST": "motivated individual",
            "EXCLUSIONARY": "qualified candidate",
            "ABILITY_CODED": "able to manage workload",
        }.get(cat, "qualified candidate")
        explanation = DEFAULT_EXPLANATIONS.get(cat, "This phrase may exclude qualified applicants.")
        return {**span, "rewrite": rewrite, "explanation": explanation}

    def _template_rewrite(self, phrase: str, category: str) -> str | None:
        """Returns a category-aware rewrite from deterministic templates when matched."""
        templates = REWRITE_TEMPLATES.get(category, {})
        if not templates:
            return None

        phrase_clean = phrase.strip()
        phrase_low = phrase_clean.lower()
        if not phrase_low:
            return None

        # Prefer exact phrase matches.
        if phrase_low in templates:
            return templates[phrase_low]

        # Fallback: replace matched token/phrase inside larger span.
        for source, target in sorted(templates.items(), key=lambda kv: len(kv[0]), reverse=True):
            pattern = rf"\b{re.escape(source)}\b"
            if re.search(pattern, phrase_low):
                rewritten = re.sub(pattern, target, phrase_clean, flags=re.IGNORECASE)
                rewritten = re.sub(r"\s+", " ", rewritten).strip()
                if rewritten and rewritten.lower() != phrase_low:
                    return rewritten
        return None

    def _call_llm(self, phrase: str, category: str, context: str) -> dict:
        prompt = USER_TEMPLATE.format(
            phrase=phrase, category=category, context=context
        )
        message = self._client.messages.create(
            model=REWRITER_MODEL,
            max_tokens=200,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
        return json.loads(raw)

    def rewrite_span(self, text: str, span: dict) -> dict:
        """Rewrites a single span. Returns the span dict enriched with rewrite + explanation."""
        template_rewrite = self._template_rewrite(span.get("text", ""), span.get("category", ""))
        template_expl = DEFAULT_EXPLANATIONS.get(
            span.get("category", ""),
            "This phrase may exclude qualified applicants.",
        )

        if self._client is None:
            return self._fallback(span)
        try:
            context = self._context_window(text, span["start"], span["end"])
            payload = self._call_llm(span["text"], span["category"], context)
            rewrite = str(payload.get("rewrite", "")).strip()
            explanation = str(payload.get("explanation", "")).strip()
            return {
                **span,
                "rewrite": rewrite or template_rewrite or span["text"],
                "explanation": explanation or template_expl,
            }
        except (json.JSONDecodeError, KeyError, Exception) as e:
            print(f"[Rewriter] LLM call failed for span '{span['text']}': {e}")
            return self._fallback(span)

    async def rewrite_all(self, text: str, spans: list[dict]) -> list[dict]:
        """
        Rewrites all spans concurrently using asyncio.
        Each call is run in a thread pool to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(None, self.rewrite_span, text, span)
            for span in spans
        ]
        return list(await asyncio.gather(*tasks))
