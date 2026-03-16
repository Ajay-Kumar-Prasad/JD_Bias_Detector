"""
rewriter.py
Uses the Anthropic API to generate a neutral rewrite and explanation
for each flagged bias span. Falls back gracefully if the API is unavailable.
"""
import os
import json
import asyncio
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
        FALLBACKS = {
            "GENDER_CODED":  ("skilled professional",   "This phrase uses gender-coded language that may discourage applicants."),
            "AGEIST":        ("motivated individual",    "This phrase implies age preference, which may exclude experienced candidates."),
            "EXCLUSIONARY":  ("expert",                  "This phrase uses exclusionary jargon that may deter underrepresented candidates."),
            "ABILITY_CODED": ("able to manage workload", "This phrase may discourage candidates with disabilities or health conditions."),
        }
        cat = span.get("category", "EXCLUSIONARY")
        rewrite, explanation = FALLBACKS.get(cat, ("qualified candidate", "This phrase may exclude qualified applicants."))
        return {**span, "rewrite": rewrite, "explanation": explanation}

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
        if self._client is None:
            return self._fallback(span)
        try:
            context = self._context_window(text, span["start"], span["end"])
            payload = self._call_llm(span["text"], span["category"], context)
            return {
                **span,
                "rewrite":     payload.get("rewrite", span["text"]),
                "explanation": payload.get("explanation", ""),
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
