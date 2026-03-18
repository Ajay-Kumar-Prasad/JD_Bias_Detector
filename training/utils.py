"""
utils.py — Metrics, logging helpers, and seqeval wrappers
"""
import numpy as np
from seqeval.metrics import classification_report
from .dataset import ID2LABEL


def compute_metrics(p):
    # Handle HF EvalPrediction object
    if hasattr(p, "predictions"):
        predictions = p.predictions
        labels = p.label_ids
    else:
        predictions, labels = p

    # Some HF models return a tuple; first element is token logits.
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # Convert logits → class ids
    preds = np.argmax(predictions, axis=2)

    true_labels = []
    true_preds = []

    for pred_row, label_row in zip(preds, labels):
        cur_preds = []
        cur_labels = []

        for pred, lab in zip(pred_row, label_row):
            if lab == -100:
                continue
            cur_preds.append(ID2LABEL.get(pred, "O"))
            cur_labels.append(ID2LABEL.get(lab, "O"))

        if cur_labels:
            true_preds.append(cur_preds)
            true_labels.append(cur_labels)

    if not true_labels:
        # Keep training/eval loop alive instead of crashing.
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "f1_gender_coded": 0.0,
            "f1_ageist": 0.0,
            "f1_exclusionary": 0.0,
            "f1_ability_coded": 0.0,
        }

    report = classification_report(
        true_labels,
        true_preds,
        output_dict=True,
        zero_division=0
    )
    micro = report.get("micro avg", {"precision": 0.0, "recall": 0.0, "f1-score": 0.0})

    # Extract entity-level F1 correctly (without B-/I- confusion)
    def get_entity_f1(entity):
        # seqeval groups B-/I- automatically under entity
        return round(report.get(entity, {}).get("f1-score", 0.0), 4)

    return {
        "precision": round(micro["precision"], 4),
        "recall":    round(micro["recall"], 4),
        "f1":        round(micro["f1-score"], 4),

        "f1_gender_coded": get_entity_f1("GENDER_CODED"),
        "f1_ageist": get_entity_f1("AGEIST"),
        "f1_exclusionary": get_entity_f1("EXCLUSIONARY"),
        "f1_ability_coded": get_entity_f1("ABILITY_CODED"),
    }


def print_classification_report(true_labels, true_preds):
    print(classification_report(true_labels, true_preds, zero_division=0))
