"""
classifier.py
Wraps the fine-tuned DeBERTa-v3 token classifier for inference.
Loaded once via dependency injection (see dependencies.py).
"""
import os
import json
import re
import torch
from transformers import pipeline, Pipeline


# Default to the cleaned v2 model for demo and API usage.
model_path = os.getenv("CLASSIFIER_MODEL_PATH", "models/deberta-jd-bias-v2-clean")
DEFAULT_THRESHOLDS = {
    "GENDER_CODED": 0.75,
    "AGEIST": 0.75,
    "ABILITY_CODED": 0.70,
    "EXCLUSIONARY": 0.75,
}
MIN_CONFIDENCE = float(os.getenv("CLASSIFIER_MIN_CONFIDENCE", "0.60"))
SINGLE_TOKEN_MIN_CONFIDENCE = float(
    os.getenv("CLASSIFIER_SINGLE_TOKEN_MIN_CONFIDENCE", "0.75")
)
STOP_WORDS = {
    w.strip().lower()
    for w in os.getenv(
        "CLASSIFIER_STOP_WORDS",
        "in,on,at,by,for,to,from,with,work,handle,level,quickly,strong",
    ).split(",")
    if w.strip()
}


def _load_thresholds() -> dict:
    raw = os.getenv("CLASSIFIER_CATEGORY_THRESHOLDS")
    if not raw:
        return DEFAULT_THRESHOLDS.copy()
    try:
        payload = json.loads(raw)
        out = DEFAULT_THRESHOLDS.copy()
        for cat, value in payload.items():
            out[str(cat)] = float(value)
        return out
    except Exception:
        print("[Classifier] Invalid CLASSIFIER_CATEGORY_THRESHOLDS; using defaults.")
        return DEFAULT_THRESHOLDS.copy()


class BiasClassifier:
    def __init__(self):
        device = 0 if torch.cuda.is_available() else -1
        self._thresholds = _load_thresholds()
        print(f"[Classifier] Loading model from '{model_path}' (device={device})")
        self._pipe: Pipeline = pipeline(
            "token-classification",
            model=model_path,
            aggregation_strategy="simple",   # merges B-/I- into one span
            device=device,
        )
        print(f"[Classifier] Category thresholds: {self._thresholds}")
        print("[Classifier] Ready.")

    @staticmethod
    def _token_count(text: str) -> int:
        return len(re.findall(r"\w+", text))

    @staticmethod
    def _can_merge(prev: dict, cur: dict, source_text: str) -> bool:
        if prev["entity_group"] != cur["entity_group"]:
            return False
        if cur["start"] < prev["end"]:
            return True
        bridge = source_text[prev["end"] : cur["start"]]
        # Merge spans separated only by whitespace/hyphen/slash.
        return bool(bridge) and all(ch.isspace() or ch in "-/" for ch in bridge)

    def _merge_adjacent_spans(self, entities: list[dict], source_text: str) -> list[dict]:
        if not entities:
            return []
        ordered = sorted(entities, key=lambda e: (e["start"], e["end"]))
        merged = [ordered[0].copy()]
        for cur in ordered[1:]:
            prev = merged[-1]
            if self._can_merge(prev, cur, source_text):
                prev["end"] = max(prev["end"], cur["end"])
                prev["score"] = max(float(prev["score"]), float(cur["score"]))
            else:
                merged.append(cur.copy())
        return merged

    def predict(self, text: str) -> list[dict]:
        """
        Run the NER pipeline on text.

        Returns:
            List of dicts:
                {
                    "text":       str,   # the biased phrase
                    "start":      int,   # char offset in original text
                    "end":        int,
                    "category":   str,   # GENDER_CODED | AGEIST | EXCLUSIONARY | ABILITY_CODED
                    "confidence": float,
                }
        """
        # Quick normalization for common split phrase.
        model_text = text.replace("fast paced", "fast-paced").replace("Fast paced", "Fast-paced")

        raw = self._pipe(model_text)
        merged = self._merge_adjacent_spans(raw, model_text)
        spans = []
        for entity in merged:
            category = entity["entity_group"]
            confidence = round(float(entity["score"]), 4)
            threshold = max(float(self._thresholds.get(category, 0.6)), MIN_CONFIDENCE)
            if confidence < threshold:
                continue
            start = int(entity["start"])
            end = int(entity["end"])
            span_text = model_text[start:end].strip()
            if not span_text:
                continue
            token_count = self._token_count(span_text)
            if token_count == 1 and confidence < SINGLE_TOKEN_MIN_CONFIDENCE:
                continue
            if token_count == 1 and span_text.lower() in STOP_WORDS:
                continue

            # Pragmatic correction for common confusion.
            if span_text.lower() in {"demanding", "intense"}:
                category = "ABILITY_CODED"

            spans.append({
                "text":       span_text,
                "start":      start,
                "end":        end,
                "category":   category,
                "confidence": confidence,
            })
        return spans
