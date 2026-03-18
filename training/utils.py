"""
utils.py — Metrics, logging helpers, and seqeval wrappers
"""
import numpy as np
from seqeval.metrics import (
    f1_score, precision_score, recall_score, classification_report
)
from .dataset import ID2LABEL


def compute_metrics(p):
    predictions, labels = p

    if predictions.shape != labels.shape:
        raise ValueError("Predictions and labels shape mismatch")

    predictions = np.argmax(predictions, axis=2)

    true_labels = [
        [ID2LABEL.get(l, "O") for l in label_row if l != -100]
        for label_row in labels
    ]

    true_preds = [
        [ID2LABEL.get(pred, "O") for pred, lab in zip(pred_row, label_row) if lab != -100]
        for pred_row, label_row in zip(predictions, labels)
    ]

    if len(true_labels) == 0:
        raise ValueError("No valid labels found after filtering")

    report = classification_report(
        true_labels,
        true_preds,
        output_dict=True,
        zero_division=0
    )

    per_class = {}
    for cat in ["GENDER_CODED", "AGEIST", "EXCLUSIONARY", "ABILITY_CODED"]:
        if cat in report:
            per_class[f"f1_{cat.lower()}"] = round(report[cat]["f1-score"], 4)
        else:
            per_class[f"f1_{cat.lower()}"] = 0.0

    return {
        "precision": round(report["micro avg"]["precision"], 4),
        "recall":    round(report["micro avg"]["recall"], 4),
        "f1":        round(report["micro avg"]["f1-score"], 4),
        **per_class,
    }


def print_classification_report(true_labels, true_preds):
    print(classification_report(true_labels, true_preds, zero_division=0))
