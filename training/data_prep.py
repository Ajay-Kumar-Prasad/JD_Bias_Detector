"""
data_prep.py — Build training data for JD Bias Detector.

Key goals:
1. Less synthetic leakage (avoid exact phrase -> exact label memorization).
2. More natural synthetic writing and lower bias density.
3. Biased/neutral balance near 50/50.
4. Optional real-JD ingestion.
"""

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

BIAS_LEXICON: Dict[str, List[str]] = {}

NEGATION_CUES = {
    "no",
    "not",
    "without",
    "never",
    "avoid",
    "avoids",
    "dont",
    "don't",
    "doesnt",
    "doesn't",
    "isnt",
    "isn't",
}

# Safety valve for known noisy labels if they appear in lexicon files.
NOISY_TERMS = {
    "AGEIST": {"high energy"},
}

ROLES = [
    "Software Engineer",
    "Data Scientist",
    "Product Manager",
    "DevOps Engineer",
    "Backend Engineer",
    "Frontend Engineer",
    "ML Engineer",
    "Technical Writer",
    "Customer Success Manager",
]

COMPANY_TYPES = [
    "a product-focused team",
    "a growing SaaS company",
    "a remote-first organization",
    "an engineering-led startup",
]

NEUTRAL_INTRO = [
    "We are hiring a {role} to join {company}.",
    "Our team is looking for a {role}.",
    "We are seeking a {role} who values collaboration and quality.",
]

NEUTRAL_RESP = [
    "You will collaborate with cross-functional partners to deliver customer value.",
    "You will design, build, and maintain reliable systems.",
    "You will communicate clearly with stakeholders and teammates.",
    "You will participate in planning, code reviews, and incident response.",
]

NEUTRAL_REQ = [
    "Experience with modern development workflows is preferred.",
    "Strong written and verbal communication skills are important.",
    "You can break down ambiguous problems into clear execution steps.",
    "You care about maintainability, testing, and documentation.",
]

BIAS_SENTENCE = [
    "We are looking for someone who can {bias}.",
    "The ideal candidate is {bias}.",
    "This role requires a {bias} mindset.",
    "You should be comfortable in a {bias} environment.",
]

HARD_NEGATIVE_NEUTRAL = [
    "We value a dynamic and inclusive work culture.",
    "Our diverse team collaborates effectively across functions.",
    "We foster a collaborative, supportive, and transparent environment.",
    "Candidates should communicate clearly and work well with others.",
    "We welcome applicants from different backgrounds and experiences.",
    "The team is dynamic, thoughtful, and focused on learning.",
]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def load_lexicons(path: str = "data/bias_lexicon") -> Dict[str, List[str]]:
    lexicon: Dict[str, List[str]] = {}
    for file in sorted(Path(path).glob("*.json")):
        with open(file) as f:
            data = json.load(f)
        category = data["category"]
        terms = [t.strip().lower() for t in data["terms"] if t.strip()]
        terms = [t for t in terms if t not in NOISY_TERMS.get(category, set())]
        lexicon[category] = sorted(set(terms))
    return lexicon


def validate_lexicon(lexicon: Dict[str, List[str]]):
    seen = {}
    for cat, terms in lexicon.items():
        for term in terms:
            if term in seen:
                raise ValueError(
                    f"Lexicon overlap detected: '{term}' in {seen[term]} and {cat}"
                )
            seen[term] = cat


def tokenize_simple(text: str) -> List[str]:
    return re.findall(r"\w[\w'-]*|[^\w\s]", text)


def clean_token(token: str) -> str:
    return re.sub(r"[^\w\-]", "", token.lower())


def _phrase_tokens(phrase: str) -> List[str]:
    return [t for t in (clean_token(x) for x in tokenize_simple(phrase)) if t]


def _is_negated(tokens: List[str], start: int, length: int) -> bool:
    left = [clean_token(t) for t in tokens[max(0, start - 4) : start]]
    right = [clean_token(t) for t in tokens[start : start + length + 6]]

    if any(tok in NEGATION_CUES for tok in left):
        return True
    if "not" in right and ("required" in right or "necessary" in right):
        return True
    return False


def annotate_iob(tokens: List[str], lexicon: Dict[str, List[str]]) -> List[str]:
    labels = ["O"] * len(tokens)
    idx_map = [i for i, tok in enumerate(tokens) if clean_token(tok)]
    cleaned = [clean_token(tokens[i]) for i in idx_map]

    for category, phrases in lexicon.items():
        parsed = sorted(
            [p for p in (_phrase_tokens(x) for x in phrases) if p],
            key=len,
            reverse=True,
        )
        for ptoks in parsed:
            n = len(ptoks)
            for i in range(len(cleaned) - n + 1):
                if cleaned[i : i + n] != ptoks:
                    continue
                span_positions = idx_map[i : i + n]
                start_pos = span_positions[0]
                if _is_negated(tokens, start_pos, n):
                    # Keep contextual negatives neutral to reduce deterministic leakage.
                    continue
                if all(labels[pos] == "O" for pos in span_positions):
                    labels[start_pos] = f"B-{category}"
                    for j in range(1, n):
                        labels[span_positions[j]] = f"I-{category}"
    return labels


def _pick_bias_phrase(
    phrase_counts: Counter,
    category_counts: Dict[str, int],
    max_phrase_repeat: int,
    max_per_category: int,
    multi_token_boost: float = 2.0,
    category: Optional[str] = None,
) -> tuple[str, str]:
    categories = [category] if category else list(BIAS_LEXICON.keys())
    random.shuffle(categories)

    for cat in categories:
        if category_counts[cat] >= max_per_category:
            continue
        pool = [p for p in BIAS_LEXICON[cat] if phrase_counts[p] < max_phrase_repeat]
        if pool:
            weights = [1.0 + (multi_token_boost if len(p.split()) > 1 else 0.0) for p in pool]
            phrase = random.choices(pool, weights=weights, k=1)[0]
            phrase_counts[phrase] += 1
            category_counts[cat] += 1
            return cat, phrase

    fallback_cats = [c for c in BIAS_LEXICON.keys() if category_counts[c] < max_per_category]
    if not fallback_cats:
        fallback_cats = list(BIAS_LEXICON.keys())
    cat = category if category and category in BIAS_LEXICON else random.choice(fallback_cats)
    phrase = random.choice(BIAS_LEXICON[cat])
    phrase_counts[phrase] += 1
    category_counts[cat] += 1
    return cat, phrase


def generate_synthetic_jd(
    biased: bool,
    phrase_counts: Counter,
    category_counts: Dict[str, int],
    max_phrase_repeat: int = 60,
    max_per_category: int = 2000,
    multi_token_boost: float = 2.0,
) -> str:
    role = random.choice(ROLES)
    company = random.choice(COMPANY_TYPES)

    lines = [random.choice(NEUTRAL_INTRO).format(role=role, company=company)]
    lines.extend(random.sample(NEUTRAL_RESP, k=2))
    lines.extend(random.sample(NEUTRAL_REQ, k=2))

    if biased:
        # Lower bias density to prevent synthetic overfitting.
        bias_budget = random.choice([1, 1, 2])
        for _ in range(bias_budget):
            _, phrase = _pick_bias_phrase(
                phrase_counts=phrase_counts,
                category_counts=category_counts,
                max_phrase_repeat=max_phrase_repeat,
                max_per_category=max_per_category,
                multi_token_boost=multi_token_boost,
            )
            lines.append(random.choice(BIAS_SENTENCE).format(bias=phrase))

        # Add occasional neutral/negated contextual usage of bias phrases.
        if random.random() < 0.35:
            _, phrase = _pick_bias_phrase(
                phrase_counts=phrase_counts,
                category_counts=category_counts,
                max_phrase_repeat=max_phrase_repeat,
                max_per_category=max_per_category,
                multi_token_boost=multi_token_boost,
            )
            lines.append(
                f"Using {phrase} language is not required for success in this role."
            )

    if random.random() < 0.4:
        lines.append("We are an equal opportunity employer.")

    return " ".join(lines)


HF_CANDIDATES = [
    ("DBD-research-group/job-descriptions-500k", "description"),
    ("swimming/job_descriptions", "job_description"),
    ("vikp/job_postings", "text"),
    ("jacob-hugging-face/job-descriptions", "job_description"),
    ("elricwan/job-description", "description"),
]


def load_hf_dataset(max_samples: int) -> List[Dict]:
    samples: List[Dict] = []
    try:
        from datasets import load_dataset
    except ImportError:
        print("   datasets package not installed. Run: pip install datasets")
        return samples

    for dataset_name, text_col in HF_CANDIDATES:
        try:
            print(f"   Trying {dataset_name} ...")
            ds = load_dataset(dataset_name, split="train")
            added = 0
            for row in ds:
                text = row.get(text_col) or ""
                if not isinstance(text, str) or len(text.split()) < 20:
                    continue
                text = text[:900].strip()
                tokens = tokenize_simple(text)
                labels = annotate_iob(tokens, BIAS_LEXICON)
                samples.append(
                    {
                        "tokens": tokens,
                        "labels": labels,
                        "text": text,
                        "source": dataset_name,
                    }
                )
                added += 1
                if len(samples) >= max_samples:
                    break
            print(f"   Added {added} samples from {dataset_name}")
        except Exception as e:
            print(f"   {dataset_name} failed: {str(e)[:80]}")
            continue
        if len(samples) >= max_samples:
            break

    return samples


def load_real_jds(path: str, max_samples: int = 500) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Real data file not found: {path}")

    texts: List[str] = []
    suffix = p.suffix.lower()

    if suffix == ".txt":
        texts = [
            line.strip()
            for line in p.read_text().splitlines()
            if len(line.split()) >= 20
        ]
    elif suffix == ".jsonl":
        with open(p) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = (
                    row.get("text")
                    or row.get("description")
                    or row.get("job_description")
                    or ""
                )
                if isinstance(text, str) and len(text.split()) >= 20:
                    texts.append(text.strip())
    elif suffix == ".csv":
        with open(p) as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = (
                    row.get("text")
                    or row.get("description")
                    or row.get("job_description")
                    or ""
                )
                if isinstance(text, str) and len(text.split()) >= 20:
                    texts.append(text.strip())
    else:
        raise ValueError("Supported real-data formats: .txt, .jsonl, .csv")

    random.shuffle(texts)
    return texts[:max_samples]


def build_sample(text: str, source: str = "synthetic") -> Dict:
    tokens = tokenize_simple(text)
    labels = annotate_iob(tokens, BIAS_LEXICON)
    return {"tokens": tokens, "labels": labels, "text": text, "source": source}


def _dominant_category(sample: Dict) -> Optional[str]:
    c = Counter()
    for lbl in sample.get("labels", []):
        if lbl.startswith("B-"):
            c[lbl.split("-", 1)[1]] += 1
    if not c:
        return None
    return c.most_common(1)[0][0]


def rebalance_biased_categories(samples: List[Dict], max_per_category: int) -> List[Dict]:
    if max_per_category <= 0:
        return samples

    biased = [s for s in samples if any(l != "O" for l in s["labels"])]
    neutral = [s for s in samples if all(l == "O" for l in s["labels"])]

    buckets: Dict[str, List[Dict]] = defaultdict(list)
    for s in biased:
        cat = _dominant_category(s)
        if cat is not None:
            buckets[cat].append(s)

    kept_biased: List[Dict] = []
    for cat, bucket in buckets.items():
        if len(bucket) > max_per_category:
            kept_biased.extend(random.sample(bucket, max_per_category))
        else:
            kept_biased.extend(bucket)

    # Keep any biased samples without dominant category as-is.
    leftover = [s for s in biased if _dominant_category(s) is None]
    result = kept_biased + leftover + neutral
    random.shuffle(result)
    return result


def generate_hard_negative_neutral(n: int) -> List[Dict]:
    out: List[Dict] = []
    for _ in range(n):
        role = random.choice(ROLES)
        company = random.choice(COMPANY_TYPES)
        intro = random.choice(NEUTRAL_INTRO).format(role=role, company=company)
        line1 = random.choice(HARD_NEGATIVE_NEUTRAL)
        line2 = random.choice(NEUTRAL_RESP)
        line3 = random.choice(NEUTRAL_REQ)
        text = " ".join([intro, line1, line2, line3])
        out.append(build_sample(text, source="synthetic_hard_negative"))
    return out


def generate_synthetic(
    n: int,
    synthetic_biased_ratio: float = 0.30,
    max_per_category: int = 2000,
    multi_token_boost: float = 2.0,
) -> List[Dict]:
    phrase_counts: Counter = Counter()
    category_counts = defaultdict(int)
    n_biased = int(n * synthetic_biased_ratio)
    n_neutral = n - n_biased
    samples: List[Dict] = []

    for _ in range(n_biased):
        samples.append(
            build_sample(
                generate_synthetic_jd(
                    True,
                    phrase_counts,
                    category_counts,
                    max_per_category=max_per_category,
                    multi_token_boost=multi_token_boost,
                ),
                "synthetic_biased",
            )
        )
    for _ in range(n_neutral):
        samples.append(
            build_sample(
                generate_synthetic_jd(
                    False,
                    phrase_counts,
                    category_counts,
                    max_per_category=max_per_category,
                    multi_token_boost=multi_token_boost,
                ),
                "synthetic_neutral",
            )
        )

    random.shuffle(samples)
    return samples


def rebalance_samples(samples: List[Dict], target_ratio: float = 0.5) -> List[Dict]:
    biased = [s for s in samples if any(l != "O" for l in s["labels"])]
    neutral = [s for s in samples if all(l == "O" for l in s["labels"])]

    if not biased or not neutral:
        return samples

    # Force exact 50/50 balance when target_ratio is 0.5 (most stable mode).
    if abs(target_ratio - 0.5) < 1e-9:
        if len(neutral) < len(biased):
            needed = len(biased) - len(neutral)
            neutral.extend(generate_hard_negative_neutral(needed))
        elif len(neutral) > len(biased):
            neutral = random.sample(neutral, len(biased))
        result = biased + neutral
        random.shuffle(result)
        return result

    max_biased = int(len(neutral) * (target_ratio / (1 - target_ratio)))
    if len(biased) > max_biased:
        biased = random.sample(biased, max_biased)

    result = biased + neutral
    random.shuffle(result)
    return result


def count_bias(samples: List[Dict]) -> dict:
    biased = sum(1 for s in samples if any(l != "O" for l in s["labels"]))
    neutral = len(samples) - biased
    return {
        "total": len(samples),
        "biased": biased,
        "neutral": neutral,
        "ratio": round((biased / len(samples)) if samples else 0.0, 4),
    }


def split_samples(
    samples: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Dict[str, List[Dict]]:
    pool = samples[:]
    random.shuffle(pool)
    n = len(pool)
    t = int(n * train_ratio)
    v = int(n * val_ratio)
    return {
        "train": pool[:t],
        "val": pool[t : t + v],
        "test": pool[t + v :],
    }


def save_splits(splits: Dict[str, List[Dict]], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ["train", "val", "test"]:
        data = splits[name]
        path = output_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for s in data:
                f.write(json.dumps(s) + "\n")
        has_bias = sum(1 for s in data if any(l != "O" for l in s["labels"]))
        print(
            f"   ✅ {name}.jsonl — {len(data)} samples "
            f"({has_bias} biased, {len(data) - has_bias} neutral) → {path}"
        )

    lexicon_dir = output_dir.parent / "bias_lexicon"
    lexicon_dir.mkdir(exist_ok=True)
    for cat, terms in BIAS_LEXICON.items():
        with open(lexicon_dir / f"{cat.lower()}.json", "w") as f:
            json.dump({"category": cat, "terms": terms}, f, indent=2)
    print(f"   ✅ Bias lexicons saved to {lexicon_dir}")


def verify_saved_splits(output_dir: Path):
    print("\n🔎 VERIFY SAVED FILES")
    for name in ["train", "val", "test"]:
        path = output_dir / f"{name}.jsonl"
        data: List[Dict] = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        stats = count_bias(data)
        print(f"{name}: {stats}")


def print_distributions(samples: List[Dict]):
    label_counts = Counter(l for s in samples for l in s["labels"])
    biased = sum(1 for s in samples if any(l != "O" for l in s["labels"]))
    neutral = len(samples) - biased
    category_counts = Counter()
    for lbl, count in label_counts.items():
        if lbl.startswith("B-"):
            category_counts[lbl.split("-", 1)[1]] += count

    print(f"\n📦 Total samples: {len(samples)}")
    print(
        f"   Split mix: biased={biased} neutral={neutral} "
        f"ratio={biased / max(1, len(samples)):.2f}"
    )
    print("   Label distribution (including O):")
    for lbl, count in sorted(label_counts.items()):
        print(f"      {lbl}: {count}")
    if category_counts:
        print("   B-tag category distribution:")
        for cat, count in sorted(category_counts.items()):
            print(f"      {cat}: {count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/annotated")
    parser.add_argument("--n_synthetic", type=int, default=1200)
    parser.add_argument("--source", choices=["hf", "synthetic", "both"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--real_data_path",
        default=None,
        help="Optional .txt/.jsonl/.csv file with real JDs",
    )
    parser.add_argument("--real_max_samples", type=int, default=300)
    parser.add_argument("--target_ratio", type=float, default=0.5)
    parser.add_argument("--synthetic_biased_ratio", type=float, default=0.30)
    parser.add_argument("--max_per_category", type=int, default=2000)
    parser.add_argument("--biased_category_cap", type=int, default=0)
    parser.add_argument("--multi_token_boost", type=float, default=2.0)
    args = parser.parse_args()

    set_seed(args.seed)

    global BIAS_LEXICON
    BIAS_LEXICON = load_lexicons()
    validate_lexicon(BIAS_LEXICON)

    output_dir = Path(args.output_dir)
    samples: List[Dict] = []

    if args.source in ("hf", "both"):
        print("⬇  Loading HuggingFace job description data ...")
        samples.extend(load_hf_dataset(max_samples=3500))

    if args.source in ("synthetic", "both"):
        needed = max(args.n_synthetic, 300 if args.source == "synthetic" else args.n_synthetic)
        print(f"⚙  Generating {needed} synthetic samples ...")
        samples.extend(
            generate_synthetic(
                needed,
                synthetic_biased_ratio=args.synthetic_biased_ratio,
                max_per_category=args.max_per_category,
                multi_token_boost=args.multi_token_boost,
            )
        )

    if args.real_data_path:
        print(f"📥 Loading real JDs from {args.real_data_path} ...")
        real_texts = load_real_jds(args.real_data_path, max_samples=args.real_max_samples)
        samples.extend(build_sample(t, "real") for t in real_texts)
        print(f"   Added {len(real_texts)} real samples")

    if args.biased_category_cap > 0:
        print(f"⚖️  Applying biased category cap: {args.biased_category_cap}")
        before_cap = len(samples)
        samples = rebalance_biased_categories(samples, max_per_category=args.biased_category_cap)
        print(f"   Samples after category cap: {len(samples)} (from {before_cap})")

    print_distributions(samples)
    splits = split_samples(samples)
    for name in ["train", "val", "test"]:
        before = count_bias(splits[name])
        print(
            f"BEFORE REBALANCE ({name}): "
            f"biased={before['biased']} neutral={before['neutral']} ratio={before['ratio']:.2f}"
        )
        splits[name] = rebalance_samples(splits[name], target_ratio=args.target_ratio)
        after = count_bias(splits[name])
        print(
            f"AFTER REBALANCE  ({name}): "
            f"biased={after['biased']} neutral={after['neutral']} ratio={after['ratio']:.2f}"
        )

    print("\n🚨 FINAL CHECK BEFORE SAVE")
    for name in ["train", "val", "test"]:
        stats = count_bias(splits[name])
        print(f"{name}: {stats}")

    save_splits(splits, output_dir)
    verify_saved_splits(output_dir)
    print("\n✅ Data preparation complete.")


if __name__ == "__main__":
    main()
