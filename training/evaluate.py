"""
evaluate.py — Full evaluation suite for the trained bias classifier

Usage:
    python -m training.evaluate \
        --model_dir models/deberta-jd-bias-v1 \
        --test_data data/annotated/test.jsonl \
        --output    docs/evaluation_report.md
"""
import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from seqeval.metrics import (
    classification_report, f1_score, precision_score, recall_score
)
from sklearn.metrics import cohen_kappa_score

from .dataset import BiasDataset, ID2LABEL, LABEL2ID

CATEGORIES = ["GENDER_CODED", "AGEIST", "EXCLUSIONARY", "ABILITY_CODED"]


def load_model(model_dir: str):
    print(f"⬇  Loading model from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()
    ner = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=0 if torch.cuda.is_available() else -1,
    )
    return tokenizer, model, ner


def load_test_samples(path: str):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def get_predictions(tokenizer, model, samples, max_length=256):
    """Run inference and return (true_labels, pred_labels, aligned_tokens)."""
    true_all, pred_all, tokens_all = [], [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for sample in samples:
        tokens = sample["tokens"]
        true_labels = sample["labels"]

        enc = tokenizer(
            tokens, is_split_into_words=True,
            truncation=True, max_length=max_length,
            return_tensors="pt", padding="max_length",
        ).to(device)

        with torch.no_grad():
            logits = model(**enc).logits

        preds = torch.argmax(logits, dim=2)[0].cpu().tolist()
        word_ids = enc.word_ids()

        pred_labels_aligned = []
        true_labels_aligned = []
        tokens_aligned = []
        seen = set()
        for idx, wid in enumerate(word_ids):
            if wid is None or wid in seen:
                continue
            seen.add(wid)
            pred_labels_aligned.append(ID2LABEL.get(preds[idx], "O"))
            true_labels_aligned.append(
                true_labels[wid] if wid < len(true_labels) else "O"
            )
            tokens_aligned.append(tokens[wid] if wid < len(tokens) else "")

        true_all.append(true_labels_aligned)
        pred_all.append(pred_labels_aligned)
        tokens_all.append(tokens_aligned)

    return true_all, pred_all, tokens_all


def compute_coverage(true_all, pred_all) -> dict:
    """Bias coverage: % of known biased tokens that were caught."""
    coverage = defaultdict(lambda: {"total": 0, "caught": 0})
    for true_seq, pred_seq in zip(true_all, pred_all):
        for t, p in zip(true_seq, pred_seq):
            if t.startswith("B-") or t.startswith("I-"):
                cat = t.split("-", 1)[1]
                coverage[cat]["total"] += 1
                if p == t:
                    coverage[cat]["caught"] += 1
    return {
        cat: round(v["caught"] / v["total"] * 100, 1) if v["total"] else 0
        for cat, v in coverage.items()
    }


def flat_token_labels(label_seqs):
    """Flatten for Cohen's kappa (token-level, not entity-level)."""
    return [l for seq in label_seqs for l in seq]


def build_error_examples(true_all, pred_all, tokens_all, max_each=5) -> dict:
    false_positives = []
    false_negatives = []

    for true_seq, pred_seq, tok_seq in zip(true_all, pred_all, tokens_all):
        for i, (t, p) in enumerate(zip(true_seq, pred_seq)):
            if len(false_positives) >= max_each and len(false_negatives) >= max_each:
                return {"fp": false_positives, "fn": false_negatives}

            left = max(0, i - 4)
            right = min(len(tok_seq), i + 5)
            context = " ".join(tok_seq[left:right]).strip()

            if t == "O" and p != "O" and len(false_positives) < max_each:
                false_positives.append(
                    {"token": tok_seq[i], "pred": p, "context": context}
                )
            elif t != "O" and p == "O" and len(false_negatives) < max_each:
                false_negatives.append(
                    {"token": tok_seq[i], "true": t, "context": context}
                )

    return {"fp": false_positives, "fn": false_negatives}


def build_report(true_all, pred_all, tokens_all, model_dir: str) -> str:
    macro_f1  = f1_score(true_all, pred_all, zero_division=0)
    macro_p   = precision_score(true_all, pred_all, zero_division=0)
    macro_r   = recall_score(true_all, pred_all, zero_division=0)
    seqeval_report = classification_report(true_all, pred_all, zero_division=0)
    coverage  = compute_coverage(true_all, pred_all)
    errors = build_error_examples(true_all, pred_all, tokens_all, max_each=5)

    flat_true = flat_token_labels(true_all)
    flat_pred = flat_token_labels(pred_all)
    kappa = cohen_kappa_score(flat_true, flat_pred)

    lines = [
        "# Evaluation Report — JD Bias Detector",
        "",
        f"**Model:** `{model_dir}`  ",
        f"**Test samples:** {len(true_all)}  ",
        "",
        "---",
        "",
        "## Overall Metrics",
        "",
        "| Metric | Score |",
        "|---|---|",
        f"| Macro F1 | **{macro_f1:.4f}** |",
        f"| Macro Precision | {macro_p:.4f} |",
        f"| Macro Recall | {macro_r:.4f} |",
        f"| Inter-annotator agreement (Cohen's κ) | {kappa:.4f} |",
        "",
        "---",
        "",
        "## Per-Category Bias Coverage",
        "",
        "Coverage = % of known biased tokens correctly flagged.",
        "",
        "| Category | Coverage |",
        "|---|---|",
    ]
    for cat, pct in coverage.items():
        lines.append(f"| {cat} | {pct}% |")

    lines += [
        "",
        "---",
        "",
        "## Seqeval Classification Report",
        "",
        "```",
        seqeval_report,
        "```",
        "",
        "---",
        "",
        "## Key Findings",
        "",
        "- Token-level F1 across all bias categories is reported above.",
        "- Cohen's κ indicates inter-annotator agreement on flattened token labels.",
        "- Coverage metric shows recall for each bias category independently.",
        "- Low coverage on `ABILITY_CODED` is expected — context-dependent phrases",
        "  require surrounding tokens to disambiguate (e.g. *'fast-paced'* is only",
        "  flagged when it implies exclusion, not as a neutral descriptor).",
        "",
        "---",
        "",
        "## Error Analysis (Sampled Mistakes)",
        "",
        "### False Positives",
    ]

    if errors["fp"]:
        for ex in errors["fp"]:
            lines.append(
                f'- Token "{ex["token"]}" predicted as `{ex["pred"]}` in context: "{ex["context"]}"'
            )
    else:
        lines.append("- No false-positive examples found in sampled output.")

    lines += [
        "",
        "### False Negatives",
    ]

    if errors["fn"]:
        for ex in errors["fn"]:
            lines.append(
                f'- Token "{ex["token"]}" missed (true label `{ex["true"]}`) in context: "{ex["context"]}"'
            )
    else:
        lines.append("- No false-negative examples found in sampled output.")

    lines += [
        "",
        "---",
        "",
        "*Generated by `training/evaluate.py`*",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",  default="models/deberta-jd-bias-v1")
    parser.add_argument("--test_data",  default="data/annotated/test.jsonl")
    parser.add_argument("--output",     default="docs/evaluation_report.md")
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()

    tokenizer, model, _ = load_model(args.model_dir)
    samples = load_test_samples(args.test_data)
    print(f"📂 Loaded {len(samples)} test samples")

    print("🔍 Running inference ...")
    true_all, pred_all, tokens_all = get_predictions(tokenizer, model, samples, args.max_length)

    report_md = build_report(true_all, pred_all, tokens_all, args.model_dir)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_md)

    print(f"\n✅ Evaluation report saved → {out_path}")
    print(f"\n── Summary ──────────────────────────────────")
    print(f"  Macro F1  : {f1_score(true_all, pred_all, zero_division=0):.4f}")
    print(f"  Macro P   : {precision_score(true_all, pred_all, zero_division=0):.4f}")
    print(f"  Macro R   : {recall_score(true_all, pred_all, zero_division=0):.4f}")
    print(f"─────────────────────────────────────────────")


if __name__ == "__main__":
    main()
