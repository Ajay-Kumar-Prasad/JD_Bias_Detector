"""
calibrate_temperature.py
Tune a single temperature parameter on validation logits to improve confidence quality.

Usage:
    python -m training.calibrate_temperature \
        --model_dir models/deberta-jd-bias-v1 \
        --val_data data/annotated/val.jsonl \
        --env_file .env
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.metrics import log_loss
from transformers import AutoModelForTokenClassification, AutoTokenizer

from .dataset import BiasDataset


def temperature_scale(logits, labels):
    logits = torch.tensor(logits, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    num_classes = logits.shape[-1]
    classes = np.arange(num_classes)

    def loss_fn(T):
        t = float(max(T[0], 1e-6))
        scaled = logits / t
        probs = torch.softmax(scaled, dim=-1).cpu().numpy()
        return log_loss(labels.cpu().numpy(), probs, labels=classes)

    result = minimize(loss_fn, x0=[1.0], bounds=[(0.1, 10.0)])
    if not result.success:
        raise RuntimeError(f"Temperature optimization failed: {result.message}")
    return float(result.x[0])


def collect_validation_logits(model, dataset, device: torch.device):
    all_logits = []
    all_labels = []

    model.eval()
    model.to(device)

    with torch.no_grad():
        for sample in dataset:
            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
            labels = sample["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(0).cpu()  # [seq_len, num_classes]

            valid_mask = labels != -100
            if valid_mask.any():
                all_logits.append(logits[valid_mask].numpy())
                all_labels.append(labels[valid_mask].numpy())

    if not all_logits:
        raise ValueError("No valid token labels found in validation set.")

    logits_np = np.concatenate(all_logits, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    return logits_np, labels_np


def update_env_temperature(env_file: Path, temperature: float):
    line = f"CLASSIFIER_CALIBRATION_TEMPERATURE={temperature:.6f}"
    if env_file.exists():
        lines = env_file.read_text().splitlines()
    else:
        lines = []

    updated = False
    for i, existing in enumerate(lines):
        if existing.startswith("CLASSIFIER_CALIBRATION_TEMPERATURE="):
            lines[i] = line
            updated = True
            break

    if not updated:
        if lines and lines[-1].strip() != "":
            lines.append("")
        lines.append("# Classifier calibration")
        lines.append(line)

    env_file.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="models/deberta-jd-bias-v1")
    parser.add_argument("--val_data", default="data/annotated/val.jsonl")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--env_file", default=".env")
    parser.add_argument("--no_write_env", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(args.model_dir):
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")
    if not os.path.isfile(args.val_data):
        raise FileNotFoundError(f"Validation dataset not found: {args.val_data}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Calibration] Device: {device}")
    print(f"[Calibration] Loading model: {args.model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    dataset = BiasDataset(args.val_data, tokenizer, max_length=args.max_length)

    print(f"[Calibration] Validation samples: {len(dataset)}")
    logits, labels = collect_validation_logits(model, dataset, device)
    print(f"[Calibration] Calibrating on {len(labels)} labeled tokens")

    best_t = temperature_scale(logits, labels)
    print(f"[Calibration] Best temperature: {best_t:.6f}")

    probs_before = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    probs_after = torch.softmax(torch.tensor(logits / best_t), dim=-1).numpy()
    classes = np.arange(logits.shape[-1])
    nll_before = log_loss(labels, probs_before, labels=classes)
    nll_after = log_loss(labels, probs_after, labels=classes)
    print(f"[Calibration] NLL before: {nll_before:.6f}")
    print(f"[Calibration] NLL after : {nll_after:.6f}")

    if not args.no_write_env:
        env_path = Path(args.env_file)
        update_env_temperature(env_path, best_t)
        print(f"[Calibration] Updated {env_path} -> CLASSIFIER_CALIBRATION_TEMPERATURE={best_t:.6f}")


if __name__ == "__main__":
    main()
