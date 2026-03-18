"""
dataset.py — PyTorch Dataset for IOB token classification
"""
import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict
import os
from collections import Counter

LABELS = [
    "O",
    "B-GENDER_CODED", "I-GENDER_CODED",
    "B-AGEIST",        "I-AGEIST",
    "B-EXCLUSIONARY",  "I-EXCLUSIONARY",
    "B-ABILITY_CODED", "I-ABILITY_CODED",
]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}
NUM_LABELS = len(LABELS)


class BiasDataset(Dataset):
    """
    Loads a .jsonl file where each line has:
        {"tokens": ["We", "need", "a", "rockstar", ...],
         "labels": ["O", "O", "O", "B-EXCLUSIONARY", ...]}
    """
    def __init__(self, path: str, tokenizer, label2id: Dict = LABEL2ID,
             max_length: int = 512):

        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")

        self.tokenizer  = tokenizer
        self.label2id   = label2id
        self.max_length = max_length

        with open(path) as f:
            self.samples = [json.loads(line) for line in f if line.strip()]

        # validate structure
        for i, s in enumerate(self.samples):
            if "tokens" not in s or "labels" not in s:
                raise ValueError(f"Invalid sample at index {i}")

            if len(s["tokens"]) != len(s["labels"]):
                raise ValueError(f"Token-label mismatch at index {i}")

        label_counts = Counter(l for s in self.samples for l in s["labels"])
        total = sum(label_counts.values())

        if total > 0 and label_counts["O"] / total > 0.98:
            print("⚠️ Warning: Dataset heavily skewed toward 'O'")

    def __len__(self):
        return len(self.samples)
    
    def __repr__(self):
        return f"BiasDataset(size={len(self.samples)}, max_length={self.max_length})"

    def __getitem__(self, idx):
        sample = self.samples[idx]
        tokens = sample["tokens"]
        raw_labels = sample["labels"]
        label2id = self.label2id

        # 🚨 HARD CHECK (fail loudly)
        if len(tokens) != len(raw_labels):
            raise ValueError(f"Token-label mismatch: {len(tokens)} vs {len(raw_labels)}")

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_offsets_mapping=False,
        )
        if len(encoding["input_ids"]) == 0:
            raise ValueError("Tokenizer produced empty input")

        if "attention_mask" not in encoding:
            raise ValueError("Tokenizer did not return attention_mask")

        word_ids = encoding.word_ids()
        if word_ids is None:
            raise ValueError("Tokenizer returned None word_ids")
        aligned_labels = []
        prev_word_id = None

        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)

            elif word_id >= len(raw_labels):   # ✅ NEW FIX
                aligned_labels.append(-100)

            elif word_id != prev_word_id:
                label_str = raw_labels[word_id]

                if label_str not in label2id:
                    raise ValueError(f"Unknown label: {label_str}")

                aligned_labels.append(label2id[label_str])

            else:
                aligned_labels.append(-100)

            prev_word_id = word_id

        return {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(aligned_labels, dtype=torch.long),
        }