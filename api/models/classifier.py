"""
classifier.py
Wraps the fine-tuned DeBERTa-v3 token classifier for inference.
Loaded once via dependency injection (see dependencies.py).
"""
import os
import torch
from transformers import pipeline, Pipeline


MODEL_PATH = os.getenv("CLASSIFIER_MODEL_PATH", "models/deberta-jd-bias-v1")


class BiasClassifier:
    def __init__(self):
        device = 0 if torch.cuda.is_available() else -1
        print(f"[Classifier] Loading model from '{MODEL_PATH}' (device={device})")
        self._pipe: Pipeline = pipeline(
            "token-classification",
            model=MODEL_PATH,
            aggregation_strategy="simple",   # merges B-/I- into one span
            device=device,
        )
        print("[Classifier] Ready.")

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
        raw = self._pipe(text)
        spans = []
        for entity in raw:
            spans.append({
                "text":       entity["word"].strip(),
                "start":      entity["start"],
                "end":        entity["end"],
                "category":   entity["entity_group"],
                "confidence": round(float(entity["score"]), 4),
            })
        return spans
