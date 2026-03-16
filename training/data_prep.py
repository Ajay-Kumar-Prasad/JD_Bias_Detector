"""
data_prep.py — Build training data for JD Bias Detector
Sources:
  1. Gaucher et al. gender-coded word lists (built-in)
  2. LinkedIn / Indeed-style synthetic JDs (generated)
  3. HuggingFace datasets: "cointegrated/jd-bias" (if available), fallback to
     "rceborg/job-descriptions" + bias_lexicon auto-labeling

Run:
    python -m training.data_prep --output_dir data/annotated --n_synthetic 1000
"""

import json
import re
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

# ─── Built-in Bias Lexicon ────────────────────────────────────────────────────

BIAS_LEXICON = {
    "GENDER_CODED": [
        # Masculine-coded
        "aggressive", "ambitious", "analytical", "assertive", "autonomous",
        "boast", "challenge", "champion", "competitive", "confident",
        "decisive", "determine", "dominant", "driven", "fearless",
        "force", "headstrong", "independent", "individual", "lead",
        "ninja", "objective", "outspoken", "rock star", "rockstar",
        "self-reliant", "self-sufficient", "stubborn", "superior",
        "crushing it", "crush it", "kill it", "dominate", "hero",
        # Feminine-coded
        "commit", "communal", "compassionate", "connect", "considerate",
        "cooperative", "dependable", "empathetic", "empower", "flexible",
        "interpersonal", "honest", "inclusive", "interdependent", "kind",
        "loyal", "nurture", "pleasant", "polite", "responsible",
        "sensitive", "support", "together", "trust", "understand",
        "warm", "yield", "enthusiastic",
    ],
    "AGEIST": [
        "young", "youthful", "young and hungry", "young and energetic",
        "recent graduate", "new graduate", "fresh graduate", "entry level",
        "digital native", "born in the digital age", "tech-savvy millennial",
        "energetic", "high energy", "young professional", "hungry",
        "up and coming", "junior", "early career",
    ],
    "EXCLUSIONARY": [
        "culture fit", "culture add", "culture match", "guy", "guys",
        "ninja", "wizard", "guru", "rockstar", "rock star", "superstar",
        "hacker", "hustler", "bro", "badass", "beast", "killer",
        "legend", "unicorn", "tribe", "fitment",
    ],
    "ABILITY_CODED": [
        "fast-paced", "high pressure", "must handle pressure",
        "must thrive under pressure", "high-stress", "demanding environment",
        "able-bodied", "must be able to stand", "physically fit",
        "must be available 24/7", "always on", "no work-life balance",
        "you live and breathe", "eat sleep breathe", "passionate to a fault",
    ],
}

# ─── IOB Annotation Helper ────────────────────────────────────────────────────

def annotate_iob(tokens: List[str], lexicon: Dict[str, List[str]]) -> List[str]:
    """Assign IOB2 labels to a token list using the lexicon."""
    labels = ["O"] * len(tokens)
    text_lower = " ".join(t.lower() for t in tokens)

    for category, phrases in lexicon.items():
        for phrase in phrases:
            phrase_tokens = phrase.lower().split()
            n = len(phrase_tokens)
            for i in range(len(tokens) - n + 1):
                window = [t.lower().strip(".,!?;:\"'") for t in tokens[i:i+n]]
                if window == phrase_tokens:
                    labels[i] = f"B-{category}"
                    for j in range(1, n):
                        labels[i+j] = f"I-{category}"
    return labels


def tokenize_simple(text: str) -> List[str]:
    """Whitespace tokenizer that preserves punctuation as separate tokens."""
    return re.findall(r"\w[\w'-]*|[^\w\s]", text)


# ─── Synthetic JD Generator ───────────────────────────────────────────────────

ROLES = [
    "Software Engineer", "Product Manager", "Data Scientist", "Marketing Manager",
    "Sales Representative", "UX Designer", "DevOps Engineer", "Business Analyst",
    "Frontend Developer", "Backend Engineer", "ML Engineer", "HR Manager",
]

NEUTRAL_PHRASES = [
    "collaborates effectively with cross-functional teams",
    "delivers high-quality work on time",
    "communicates clearly with stakeholders",
    "solves complex problems",
    "contributes to team goals",
    "demonstrates strong technical skills",
    "has a growth mindset",
    "takes ownership of deliverables",
    "works well in a collaborative setting",
    "manages multiple priorities effectively",
]

BIASED_PHRASES = [p for phrases in BIAS_LEXICON.values() for p in phrases]

TEMPLATES = [
    "We are looking for a {role} who is {trait1} and {trait2}.",
    "Join our team as a {role}. You should be {trait1}, {trait2}, and ready to {trait3}.",
    "We need a {trait1} {role} to {trait3} in our {trait2} environment.",
    "As a {role}, you will {trait3}. The ideal candidate is {trait1} and {trait2}.",
    "We are hiring a {role}. You must be {trait1} with a {trait2} attitude and {trait3}.",
]

def generate_synthetic_jd(biased: bool = True) -> str:
    role = random.choice(ROLES)
    if biased:
        t1 = random.choice(BIASED_PHRASES)
        t2 = random.choice(BIASED_PHRASES)
        t3 = random.choice(BIASED_PHRASES)
    else:
        t1 = random.choice(NEUTRAL_PHRASES)
        t2 = random.choice(NEUTRAL_PHRASES)
        t3 = random.choice(NEUTRAL_PHRASES)

    template = random.choice(TEMPLATES)
    return template.format(role=role, trait1=t1, trait2=t2, trait3=t3)


# ─── Dataset Builder ──────────────────────────────────────────────────────────

def build_sample(text: str) -> Dict:
    tokens = tokenize_simple(text)
    labels = annotate_iob(tokens, BIAS_LEXICON)
    return {"tokens": tokens, "labels": labels, "text": text}


def load_hf_dataset(split_sizes: Tuple[int, int, int]) -> List[Dict]:
    """
    Try loading from HuggingFace Hub.
    Falls back to synthetic generation if dataset unavailable.
    """
    samples = []
    try:
        from datasets import load_dataset as hf_load
        print("⬇  Trying HuggingFace dataset: rceborg/job-descriptions ...")
        ds = hf_load("rceborg/job-descriptions", split="train", trust_remote_code=True)
        for row in ds:
            text = row.get("description") or row.get("text") or ""
            if len(text.split()) > 20:
                samples.append(build_sample(text[:1000]))  # cap length
            if len(samples) >= sum(split_sizes):
                break
        print(f"   Loaded {len(samples)} samples from HuggingFace.")
    except Exception as e:
        print(f"   HuggingFace load failed ({e}). Using synthetic data.")

    return samples


def generate_synthetic(n: int) -> List[Dict]:
    samples = []
    for i in range(n):
        biased = random.random() > 0.35          # ~65% biased, 35% neutral
        text = generate_synthetic_jd(biased)
        samples.append(build_sample(text))
    return samples


def split_and_save(samples: List[Dict], output_dir: Path,
                   train_ratio=0.8, val_ratio=0.1):
    random.shuffle(samples)
    n = len(samples)
    t = int(n * train_ratio)
    v = int(n * val_ratio)

    splits = {
        "train": samples[:t],
        "val":   samples[t:t+v],
        "test":  samples[t+v:],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in splits.items():
        path = output_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for sample in data:
                f.write(json.dumps(sample) + "\n")
        print(f"   ✅ {name}.jsonl — {len(data)} samples → {path}")

    # Save lexicon snapshots
    lexicon_dir = output_dir.parent / "bias_lexicon"
    lexicon_dir.mkdir(exist_ok=True)
    for cat, terms in BIAS_LEXICON.items():
        with open(lexicon_dir / f"{cat.lower()}.json", "w") as f:
            json.dump({"category": cat, "terms": terms}, f, indent=2)
    print(f"   ✅ Bias lexicons saved to {lexicon_dir}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",   default="data/annotated")
    parser.add_argument("--n_synthetic",  type=int, default=1000,
                        help="Synthetic samples if HF dataset unavailable")
    parser.add_argument("--source",       choices=["hf", "synthetic", "both"],
                        default="both")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    samples = []

    if args.source in ("hf", "both"):
        hf_samples = load_hf_dataset((800, 100, 100))
        samples.extend(hf_samples)

    if args.source in ("synthetic", "both") or len(samples) < 500:
        needed = max(args.n_synthetic, 500 - len(samples))
        print(f"⚙  Generating {needed} synthetic samples ...")
        samples.extend(generate_synthetic(needed))

    print(f"\n📦 Total samples: {len(samples)}")
    split_and_save(samples, output_dir)
    print("\n✅ Data preparation complete.")


if __name__ == "__main__":
    main()
