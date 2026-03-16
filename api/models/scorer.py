"""
scorer.py
Computes an inclusivity score (0–100) and per-category breakdown.

Scoring logic:
  - Start at 100
  - Each flagged span deducts: weight[category] × confidence × length_factor
  - length_factor = log(word_count) / log(200) so longer JDs aren't unfairly penalised
  - Score is clamped to [0, 100] and rounded to int
"""
import math


CATEGORY_WEIGHTS = {
    "GENDER_CODED":  12,
    "AGEIST":        18,
    "EXCLUSIONARY":  10,
    "ABILITY_CODED": 15,
}

CATEGORIES = list(CATEGORY_WEIGHTS.keys())


class BiasScorer:
    def score(self, text: str, spans: list[dict]) -> tuple[int, dict]:
        """
        Args:
            text:  Original job description text.
            spans: List of enriched span dicts (with category + confidence).

        Returns:
            (inclusivity_score: int, breakdown: dict[category, count])
        """
        word_count   = max(len(text.split()), 1)
        length_norm  = math.log(max(word_count, 2)) / math.log(200)

        penalty = 0.0
        for span in spans:
            weight   = CATEGORY_WEIGHTS.get(span["category"], 10)
            penalty += weight * span["confidence"] * length_norm

        score = max(0, min(100, round(100 - penalty)))

        breakdown = {cat: 0 for cat in CATEGORIES}
        for span in spans:
            cat = span.get("category")
            if cat in breakdown:
                breakdown[cat] += 1

        return score, breakdown
