"""
data_prep.py — Build training data for JD Bias Detector

HuggingFace sources (tried in order, first success wins):
  1. lukebarousse/data_nerd_jobs          — real tech JDs, public
  2. elricwan/job-description             — general JDs, public
  3. Mxode/job-description-ner           — pre-labeled NER dataset
  4. jovialjoy/job-postings-2023         — scraped postings
  Fallback: rich synthetic generation (runs offline, no HF needed)

Usage:
    python -m training.data_prep --output_dir data/annotated --n_synthetic 1000
    python -m training.data_prep --source synthetic   # offline mode
    python -m training.data_prep --source hf          # HF only
"""

import json
import re
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

# ─── Bias Lexicon ─────────────────────────────────────────────────────────────

BIAS_LEXICON = {
    "GENDER_CODED": [
        "aggressive", "assertive", "dominant", "competitive",
        "fearless", "ambitious", "driven", "results-driven",
        "self-starter", "go-getter", "high achiever",
        "decisive", "bold", "tenacious",

        # 🔥 phrase-level
        "crush it", "kill it", "dominate", "take ownership",
    ],

    "AGEIST": [
        "young", "youthful", "young and hungry", "young and energetic",
        "recent graduate", "new graduate", "fresh graduate",
        "digital native", "tech-savvy millennial",
        "young professional", "up and coming",
        "early career", "fresh out of school", "just graduated",
        "young talent", "next generation", "millennial mindset",
        "recent grad welcome", "no experience necessary",
        "entry level mindset", "fresh perspective", "new to industry ok",
    ],

    "EXCLUSIONARY": [
        "rockstar", "rock star", "rockstar engineer",
        "ninja", "ninja-level", "ninja-level skills",
        "wizard", "guru", "superstar",
        "hustler", "10x engineer", "unicorn",

        "culture fit", "culture match", "culture add",
        "culture first", "fits our culture",
        "startup dna", "we work hard we play hard",
    ],

    "ABILITY_CODED": [
        "fast-paced", "fast-paced environment",
        "high pressure", "high pressure environment",
        "must handle pressure", "must thrive under pressure",
        "high stress", "high stress environment",

        # 🔥 CRITICAL missing ones
        "demanding environment",
        "demanding workload",
        "intense challenges",

        "always on", "24/7 availability",
        "no work-life balance",
        "on call", "on call at all times",
        "available nights and weekends",

        "relentless pace",
        "sink or swim", "trial by fire", "baptism by fire",

        "thick skin required",
        "handle ambiguity", "thrive in chaos",

        "wear many hats",
        "roll up your sleeves",
        "no hand-holding",
        "hit the ground running",
    ]
}
# ─── Lexicon Validation (NEW) ────────────────────────────────────────────────

def validate_lexicon(lexicon):
    seen = {}
    for cat, terms in lexicon.items():
        for term in terms:
            key = term.lower()
            if key in seen:
                raise ValueError(
                    f"Lexicon overlap detected: '{term}' in {seen[key]} and {cat}"
                )
            seen[key] = cat

validate_lexicon(BIAS_LEXICON)

# ─── IOB Annotation ───────────────────────────────────────────────────────────

def tokenize_simple(text: str) -> List[str]:
    return re.findall(r"\w[\w'-]*|[^\w\s]", text)


def clean_token(t):
    return re.sub(r"[^\w\-]", "", t.lower())

def annotate_iob(tokens: List[str], lexicon: Dict[str, List[str]]) -> List[str]:
    labels = ["O"] * len(tokens)

    for category, phrases in lexicon.items():
        for phrase in sorted(phrases, key=len, reverse=True):
            phrase_tokens = phrase.lower().split()
            n = len(phrase_tokens)

            for i in range(len(tokens) - n + 1):
                window = [clean_token(t) for t in tokens[i:i+n]]

                if window == phrase_tokens:
                    if all(labels[i+j] == "O" for j in range(n)):
                        labels[i] = f"B-{category}"
                        for j in range(1, n):
                            labels[i+j] = f"I-{category}"

    return labels

# ─── Rich Synthetic Generator ─────────────────────────────────────────────────

ROLES = [
    "Software Engineer", "Senior Software Engineer", "Product Manager",
    "Data Scientist", "Marketing Manager", "Sales Representative",
    "UX Designer", "DevOps Engineer", "Business Analyst",
    "Frontend Developer", "Backend Engineer", "ML Engineer",
    "Full Stack Developer", "Data Engineer", "Cloud Architect",
    "Security Engineer", "QA Engineer", "Technical Lead",
    "Engineering Manager", "Product Designer", "Growth Manager",
    "Customer Success Manager", "Operations Manager", "HR Manager",
]

COMPANIES = [
    "our team", "our startup", "our engineering team",
    "our growing company", "our product team", "our fast-growing team",
]

NEUTRAL_OPENERS = [
    "We are seeking a {role} to join {company}.",
    "We are looking for an experienced {role}.",
    "{company} is hiring a {role}.",
    "Join {company} as a {role}.",
    "We have an exciting opportunity for a {role}.",
    "We are expanding and looking for a skilled {role}.",
]

BIASED_OPENERS = [
    "We are looking for a {bias1} {role} who is {bias2} and ready to {bias3}.",
    "Join {company} as a {role}. You must be {bias1} and {bias2}.",
    "We need a {bias1} {role} to {bias3} in our {bias2} environment.",
    "As a {role}, you will {bias3}. The ideal candidate is {bias1} and {bias2}.",
    "We are hiring a {bias1}, {bias2} {role} who can {bias3}.",
    "Looking for a {role} who is {bias1} with a {bias2} attitude.",
]

NEUTRAL_REQUIREMENTS = [
    "You will collaborate effectively with cross-functional teams.",
    "You will deliver high-quality work on time.",
    "You will communicate clearly with stakeholders.",
    "Strong problem-solving skills are required.",
    "You will contribute to team goals and shared objectives.",
    "Experience with modern development practices is preferred.",
    "You will take ownership of your deliverables.",
    "A growth mindset and willingness to learn are valued.",
    "You will work in a collaborative, inclusive environment.",
    "We offer competitive compensation and flexible working arrangements.",
    "You will mentor junior team members.",
    "Strong written and verbal communication skills required.",
    "Experience working in cross-functional teams preferred.",
    "You will participate in code reviews and technical discussions.",
]

BIASED_REQUIREMENTS = [
    "You must {bias} to succeed in this role.",
    "The ideal candidate is {bias} and ready to hit the ground running.",
    "We are looking for someone who is {bias} and can thrive in our culture.",
    "You should be {bias} with a proven track record.",
    "Candidates must be {bias} and comfortable in a {bias2} environment.",
    "We expect our team members to be {bias}.",
]
BIASED_PHRASES = [
    "young and hungry",
    "fast-paced environment",
    "demanding environment",
    "intense challenges",
    "rockstar engineer",
    "ninja-level skills",
    "hit the ground running",
    "crush it",
    "work hard play hard",
]
NEUTRAL_HARD = [
    "collaborative team player",
    "strong communication skills",
    "supports colleagues effectively",
    "works well with others",
    "team-oriented environment",
    "effective cross-functional collaboration",   # ✅ NEW
]


def pick_bias(category: str = None) -> str:
    if category:
        return random.choice(BIAS_LEXICON[category])
    cat = random.choice(list(BIAS_LEXICON.keys()))
    return random.choice(BIAS_LEXICON[cat])


def generate_synthetic_jd(biased: bool = True) -> str:
    role    = random.choice(ROLES)
    company = random.choice(COMPANIES)

    parts = []

    # ─── 🔥 STRONG PHRASE SIGNAL (ALWAYS for biased) ───
    if biased:
        phrase = random.choice(BIASED_PHRASES)
        parts.append(f"We are looking for someone who thrives in a {phrase}.")

        # Add second bias phrase sometimes (multi-pattern learning)
        if random.random() < 0.5:
            phrase2 = random.choice(BIASED_PHRASES)
            parts.append(f"You will thrive in a {phrase2}.")

        # 🔁 reinforce phrase (important for learning)
        if random.random() < 0.3:
            parts.append(f"The role involves working in a {phrase}.")

    # ─── STRUCTURED JD FORMAT ───
    if biased:
        opener = random.choice(BIASED_OPENERS).format(
            role=role,
            company=company,
            bias1=pick_bias(),
            bias2=pick_bias(),
            bias3=pick_bias(),
        )
        parts.append(opener)
        parts.append("About the role:")

        # Section: Responsibilities
        parts.append("Responsibilities:")
        for _ in range(random.randint(1, 2)):
            parts.append(random.choice(NEUTRAL_REQUIREMENTS))

        # Section: Requirements
        parts.append("Requirements:")

        n_bias_lines = random.choice([3, 4])
        for _ in range(n_bias_lines):
            req = random.choice(BIASED_REQUIREMENTS).format(
                bias=pick_bias(),
                bias2=pick_bias()
            )
            parts.append(req)

    else:
        opener = random.choice(NEUTRAL_OPENERS).format(
            role=role,
            company=company
        )
        parts.append(opener)

        # Mix both types properly
        for _ in range(2):
            parts.append(random.choice(NEUTRAL_REQUIREMENTS))

        for _ in range(2):
            parts.append(random.choice(NEUTRAL_HARD))

    return " ".join(parts)


# ─── HuggingFace Loader ───────────────────────────────────────────────────────

# HF_CANDIDATES = [
#     # (dataset_name, text_column)
#     ("lukebarousse/data_nerd_jobs",     "job_description"),
#     ("elricwan/job-description",        "description"),
#     ("Mxode/job-description-ner",       "text"),
#     ("jovialjoy/job-postings-2023",     "description"),
#     ("jacob-hugging-face/job-descriptions", "job_description"),
# ]
HF_CANDIDATES = [
    ("DBD-research-group/job-descriptions-500k", "description"),
    ("swimming/job_descriptions",                "job_description"),
    ("vikp/job_postings",                        "text"),
    ("jacob-hugging-face/job-descriptions",      "job_description"),
    ("elricwan/job-description",                 "description"),
]


def load_hf_dataset(max_samples: int) -> List[Dict]:
    samples = []
    try:
        from datasets import load_dataset
    except ImportError:
        print("   datasets package not installed. Run: pip install datasets")
        return samples

    for dataset_name, text_col in HF_CANDIDATES:
        try:
            print(f"   Trying {dataset_name} ...")
            ds = load_dataset(dataset_name, split="train")
            for row in ds:
                text = row.get(text_col) or ""
                if not isinstance(text, str) or len(text.split()) < 15:
                    continue
                text = text[:800].strip()
                tokens = tokenize_simple(text)
                labels = annotate_iob(tokens, BIAS_LEXICON)
                samples.append({
                    "tokens": tokens,
                    "labels": labels,
                    "text":   text,
                    "source": dataset_name,
                })
                if len(samples) >= max_samples:
                    break
            print(f"   Loaded {len(samples)} samples from {dataset_name}.")
            return samples
        except Exception as e:
            print(f"   {dataset_name} failed: {str(e)[:80]}")
            continue

    print("   All HuggingFace sources failed. Falling back to synthetic.")
    return samples


# ─── Dataset Builder ──────────────────────────────────────────────────────────

def build_sample(text: str, source: str = "synthetic") -> Dict:
    tokens = tokenize_simple(text)
    labels = annotate_iob(tokens, BIAS_LEXICON)
    return {"tokens": tokens, "labels": labels, "text": text, "source": source}


def generate_synthetic(n: int) -> List[Dict]:
    samples = []
    # n_biased  = int(n * 0.65)
    # NEW — 50/50 split, less synthetic dominance
    n_biased  = int(n * 0.50)
    n_neutral = n - n_biased
    for _ in range(n_biased):
        samples.append(build_sample(generate_synthetic_jd(biased=True),  "synthetic_biased"))
    for _ in range(n_neutral):
        samples.append(build_sample(generate_synthetic_jd(biased=False), "synthetic_neutral"))
    random.shuffle(samples)
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
            for s in data:
                f.write(json.dumps(s) + "\n")
        has_bias = sum(1 for s in data if any(l != "O" for l in s["labels"]))
        print(f"   ✅ {name}.jsonl — {len(data)} samples "
              f"({has_bias} biased, {len(data)-has_bias} neutral) → {path}")

    # Save lexicon snapshots
    lexicon_dir = output_dir.parent / "bias_lexicon"
    lexicon_dir.mkdir(exist_ok=True)
    for cat, terms in BIAS_LEXICON.items():
        p = lexicon_dir / f"{cat.lower()}.json"
        if not p.exists():
            with open(p, "w") as f:
                json.dump({"category": cat, "terms": terms}, f, indent=2)
    print(f"   ✅ Bias lexicons saved to {lexicon_dir}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",  default="data/annotated")
    parser.add_argument("--n_synthetic", type=int, default=1000,
                        help="Synthetic samples to generate (used as fallback or addition)")
    parser.add_argument("--source", choices=["hf", "synthetic", "both"],
                        default="both",
                        help="'hf' = HuggingFace only | 'synthetic' = offline | 'both' = HF + synthetic")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    samples    = []

    if args.source in ("hf", "both"):
        print("⬇  Trying HuggingFace datasets ...")
        hf_samples = load_hf_dataset(max_samples=3500)
        samples.extend(hf_samples)

    if args.source in ("synthetic", "both") or len(samples) < 300:
        needed = max(args.n_synthetic, 300 - len(samples))
        print(f"⚙  Generating {needed} synthetic samples ...")
        samples.extend(generate_synthetic(needed))

    print(f"\n📦 Total samples: {len(samples)}")

    # Print label distribution
    from collections import Counter
    label_counts = Counter(
    l for s in samples for l in s["labels"]
)

    print("   Label distribution (including O):")
    for lbl, count in sorted(label_counts.items()):
        print(f"      {lbl}: {count}")

    split_and_save(samples, output_dir)
    print("\n✅ Data preparation complete.")


if __name__ == "__main__":
    main()