"""
classifier.py
Wraps the fine-tuned DeBERTa-v3 token classifier for inference.
Loaded once via dependency injection (see dependencies.py).
"""
import os
import json
import math
import torch
from transformers import (
    pipeline,
    Pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


MODEL_NAME = "Ajay-Kumar-Prasad/jd-bias-detector"
DEFAULT_THRESHOLDS = {
    "GENDER_CODED": 0.75,
    "AGEIST": 0.75,
    "ABILITY_CODED": 0.70,
    "EXCLUSIONARY": 0.75,
}
MIN_CONFIDENCE = float(os.getenv("CLASSIFIER_MIN_CONFIDENCE", "0.60"))
CALIBRATION_TEMPERATURE = max(float(os.getenv("CLASSIFIER_CALIBRATION_TEMPERATURE", "1.0")), 1e-6)
NEUTRAL_LABELS = {"NEUTRAL", "NON_BIASED", "NOT_BIASED", "O", "LABEL_0"}


def _apply_temperature_scaling(prob: float, temperature: float) -> float:
    """
    Calibrate confidence using temperature scaling on the logit space.
    T > 1.0 softens confidence, T < 1.0 sharpens confidence.
    """
    p = min(max(float(prob), 1e-6), 1 - 1e-6)
    logit = math.log(p / (1 - p))
    calibrated = 1 / (1 + math.exp(-(logit / temperature)))
    return float(calibrated)


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
        print(f"[Classifier] Loading model from '{MODEL_NAME}' (device={device})")
        print(f"[Classifier] Calibration temperature: {CALIBRATION_TEMPERATURE}")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self._pipe: Pipeline = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
            top_k=None,
        )
        print(f"[Classifier] Category thresholds: {self._thresholds}")
        print("[Classifier] Ready.")

    @staticmethod
    def _normalize_label(raw_label: str) -> str:
        label = str(raw_label).upper().strip()
        for prefix in ("B-", "I-", "S-", "E-", "L-", "U-"):
            if label.startswith(prefix):
                label = label[len(prefix):]
        return label

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

        predictions = self._pipe(model_text, truncation=True)
        if not predictions:
            return []

        if isinstance(predictions[0], list):
            candidates = predictions[0]
        else:
            candidates = predictions

        ranked = sorted(candidates, key=lambda p: float(p.get("score", 0.0)), reverse=True)
        for candidate in ranked:
            normalized = self._normalize_label(candidate.get("label", ""))
            confidence = round(
                _apply_temperature_scaling(float(candidate.get("score", 0.0)), CALIBRATION_TEMPERATURE),
                4,
            )
            if normalized in NEUTRAL_LABELS:
                continue
            threshold = max(float(self._thresholds.get(normalized, 0.6)), MIN_CONFIDENCE)
            if confidence < threshold:
                continue
            category = normalized if normalized in DEFAULT_THRESHOLDS else "EXCLUSIONARY"
            return [{
                "text": model_text.strip(),
                "start": 0,
                "end": len(model_text),
                "category": category,
                "confidence": confidence,
            }]

        return []
