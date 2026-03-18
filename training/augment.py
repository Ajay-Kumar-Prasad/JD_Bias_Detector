"""
augment.py — Counterfactual data augmentation for the bias dataset

Strategy:
  1. Gender swap   — swap masculine↔feminine coded terms in existing samples
  2. Label-preserving paraphrase — replace neutral context words, keep bias spans
  3. Negative mining — generate hard negatives (bias word used neutrally in context)

Usage:
    python -m training.augment \
        --input  data/annotated/train.jsonl \
        --output data/annotated/train_augmented.jsonl \
        --factor 2
"""
import json
import random
import argparse
import re
from pathlib import Path
from typing import List, Dict

# ─── Swap tables ─────────────────────────────────────────────────────────────

GENDER_SWAP = {
    "he": "she", "she": "he",
    "him": "her", "her": "him",
    "his": "their", "hers": "theirs",
    "man": "woman", "woman": "man",
    "men": "women", "women": "men",
    "guy": "person", "guys": "people",
    "aggressive": "assertive",
    "dominant": "collaborative",
    "rockstar": "expert",
    "ninja": "specialist",
    "nurturing": "supportive",
    "compassionate": "empathetic",
}

CONTEXT_SYNONYMS = {
    "looking for":   ["seeking", "searching for", "hiring"],
    "join our":      ["be part of our", "come work with our"],
    "team":          ["group", "department", "unit"],
    "environment":   ["setting", "workplace", "culture"],
    "opportunity":   ["role", "position", "opening"],
    "experience":    ["background", "expertise", "track record"],
    "skills":        ["abilities", "competencies", "capabilities"],
}


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w[\w'-]*|[^\w\s]", text)


def swap_gender_tokens(tokens: List[str], labels: List[str]) -> Dict:
    new_tokens = []
    new_labels = []

    for tok, lbl in zip(tokens, labels):
        lower = tok.lower()
        swapped = GENDER_SWAP.get(lower, tok)

        if tok[0].isupper():
            swapped = swapped.capitalize()

        new_tokens.append(swapped)

        # If word changed meaning → neutralize label
        if swapped != tok:
            new_labels.append("O")
        else:
            new_labels.append(lbl)

    return {
        "tokens": new_tokens,
        "labels": new_labels,
        "text": " ".join(new_tokens),
        "augmentation": "gender_swap"
    }


def paraphrase_context(tokens: List[str], labels: List[str]) -> Dict:
    new_tokens = tokens[:]
    new_labels = labels[:]

    i = 0
    while i < len(new_tokens):
        if new_labels[i] != "O":
            i += 1
            continue

        for phrase, synonyms in CONTEXT_SYNONYMS.items():
            phrase_toks = phrase.split()
            n = len(phrase_toks)

            window = [t.lower() for t in new_tokens[i:i+n]]

            if window == phrase_toks:
                replacement = random.choice(synonyms).split()

                # replace tokens
                new_tokens[i:i+n] = replacement

                # adjust labels safely
                new_labels[i:i+n] = ["O"] * n
                if len(replacement) > n:
                    new_labels[i:i] = ["O"] * (len(replacement) - n)
                elif len(replacement) < n:
                    del new_labels[i+len(replacement):i+n]

                i += len(replacement)
                break
        else:
            i += 1

    return {
        "tokens": new_tokens,
        "labels": new_labels,
        "text": " ".join(new_tokens),
        "augmentation": "paraphrase"
    }


def hard_negative(tokens: List[str], labels: List[str]) -> Dict:
    NEGATIONS = [
        "We do not require",
        "This role does not demand",
        "You don't need to be",
        "No need to be"
    ]

    new_tokens = tokens[:]
    new_labels = labels[:]

    for i, lbl in enumerate(labels):
        if lbl.startswith("B-"):
            negation = random.choice(NEGATIONS).split()

            # insert negation BEFORE bias phrase
            new_tokens = new_tokens[:i] + negation + new_tokens[i:]

            # shift labels
            new_labels = (
                new_labels[:i]
                + ["O"] * len(negation)
                + ["O" if l != "O" else "O" for l in new_labels[i:]]
            )
            break

    return {
        "tokens": new_tokens,
        "labels": new_labels,
        "text": " ".join(new_tokens),
        "augmentation": "hard_negative"
    }


STRATEGIES = [swap_gender_tokens, paraphrase_context, hard_negative]


def augment_samples(samples: List[Dict], factor: int = 2) -> List[Dict]:
    augmented = []
    for sample in samples:
        tokens = sample["tokens"]
        labels = sample["labels"]
        has_bias = any(l != "O" for l in labels)

        # Always keep original
        augmented.append(sample)

        # Only augment samples that have at least one bias label
        if not has_bias:
            if random.random() < 0.3:
                try:
                    new_sample = paraphrase_context(tokens[:], labels[:])
                    augmented.append(new_sample)
                except:
                    pass
            continue

        for i in range(factor - 1):
            strategy = STRATEGIES[i % len(STRATEGIES)]
            try:
                new_sample = strategy(tokens[:], labels[:])
                augmented.append(new_sample)
            except Exception:
                pass    # skip if augmentation fails for this sample

    random.shuffle(augmented)
    return augmented


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/annotated/train.jsonl")
    parser.add_argument("--output", default="data/annotated/train_augmented.jsonl")
    parser.add_argument("--factor", type=int, default=2,
                        help="Augmentation factor (total copies per biased sample)")
    args = parser.parse_args()

    with open(args.input) as f:
        samples = [json.loads(l) for l in f if l.strip()]

    print(f"📂 Loaded {len(samples)} original samples")
    augmented = augment_samples(samples, args.factor)
    print(f"✅ Augmented to {len(augmented)} samples (factor={args.factor})")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for s in augmented:
            f.write(json.dumps(s) + "\n")

    print(f"💾 Saved → {args.output}")


if __name__ == "__main__":
    main()
