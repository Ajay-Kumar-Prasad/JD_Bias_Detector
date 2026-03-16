"""
dataset.py — PyTorch Dataset for IOB token classification
"""
import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict

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
        self.tokenizer  = tokenizer
        self.label2id   = label2id
        self.max_length = max_length

        with open(path) as f:
            self.samples = [json.loads(line) for line in f if line.strip()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        tokens = sample["tokens"]
        raw_labels = sample["labels"]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_offsets_mapping=False,
        )

        # Align labels to subword tokens
        word_ids = encoding.word_ids()
        aligned_labels = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)          # special tokens → ignore
            elif word_id != prev_word_id:
                label_str = raw_labels[word_id] if word_id < len(raw_labels) else "O"
                aligned_labels.append(self.label2id.get(label_str, 0))
            else:
                # Continuation subword: use I- label if B- was assigned, else -100
                label_str = raw_labels[word_id] if word_id < len(raw_labels) else "O"
                if label_str.startswith("B-"):
                    aligned_labels.append(self.label2id.get("I-" + label_str[2:], -100))
                else:
                    aligned_labels.append(-100)
            prev_word_id = word_id

        encoding["labels"] = aligned_labels
        return {k: torch.tensor(v) for k, v in encoding.items()}
