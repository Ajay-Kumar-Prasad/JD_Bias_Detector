"""
utils.py — Metrics, logging helpers, and seqeval wrappers
"""
import numpy as np
from seqeval.metrics import (
    f1_score, precision_score, recall_score, classification_report
)
from .dataset import ID2LABEL


def compute_metrics(p):
    """
    HuggingFace Trainer-compatible metric function.
    Returns macro precision, recall, F1 + per-class F1.
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [
        [ID2LABEL[l] for l in label_row if l != -100]
        for label_row in labels
    ]
    true_preds = [
        [ID2LABEL[pred] for pred, label in zip(pred_row, label_row) if label != -100]
        for pred_row, label_row in zip(predictions, labels)
    ]

    report = classification_report(true_labels, true_preds, output_dict=True, zero_division=0)

    # Per-class F1 for the four bias categories
    per_class = {}
    for cat in ["GENDER_CODED", "AGEIST", "EXCLUSIONARY", "ABILITY_CODED"]:
        key = f"B-{cat}"
        if key in report:
            per_class[f"f1_{cat.lower()}"] = round(report[key]["f1-score"], 4)

    return {
        "precision": round(precision_score(true_labels, true_preds, zero_division=0), 4),
        "recall":    round(recall_score(true_labels, true_preds, zero_division=0), 4),
        "f1":        round(f1_score(true_labels, true_preds, zero_division=0), 4),
        **per_class,
    }


def print_classification_report(true_labels, true_preds):
    print(classification_report(true_labels, true_preds, zero_division=0))
