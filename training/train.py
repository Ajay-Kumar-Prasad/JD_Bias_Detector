"""
train.py — Fine-tune DeBERTa-v3-base for JD bias token classification

Usage (cloud GPU):
    python -m training.train --config training/configs/deberta_base.yaml

Colab/Kaggle one-liner:
    !python -m training.train --config training/configs/deberta_base.yaml
"""
import argparse
import yaml
import torch
from pathlib import Path
import json
from collections import Counter
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)
from .dataset import BiasDataset, LABEL2ID, ID2LABEL, NUM_LABELS
from .utils import compute_metrics
from torch.nn import CrossEntropyLoss


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def compute_class_weights(path, label2id):
    counts = Counter()

    with open(path) as f:
        for line in f:
            if line.strip():
                counts.update(json.loads(line)["labels"])

    total = sum(counts.values())
    weights = torch.ones(len(label2id))

    for lbl, idx in label2id.items():
        if lbl == "O":
            continue
        weights[idx] = min(10.0, total / max(1, counts[lbl]))

    return weights


def _split_stats(path: str) -> dict:
    total = 0
    biased = 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)
            total += 1
            if any(lbl != "O" for lbl in sample.get("labels", [])):
                biased += 1
    return {"total": total, "biased": biased, "neutral": total - biased}


def validate_data_splits(train_path: str, val_path: str, test_path: str):
    for p in [train_path, val_path]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Required split missing: {p}")
        if Path(p).stat().st_size == 0:
            raise ValueError(f"Required split is empty: {p}")

    train_stats = _split_stats(train_path)
    val_stats = _split_stats(val_path)

    if train_stats["total"] == 0:
        raise ValueError("Train split has zero samples.")
    if val_stats["total"] == 0:
        raise ValueError("Val split has zero samples.")
    if train_stats["biased"] == 0:
        raise ValueError("Train split has no biased examples.")
    if train_stats["neutral"] == 0:
        raise ValueError("Train split has no neutral examples.")

    print(
        "📊 Split sanity:"
        f" train={train_stats['total']} (biased={train_stats['biased']}, neutral={train_stats['neutral']})"
        f" | val={val_stats['total']} (biased={val_stats['biased']}, neutral={val_stats['neutral']})"
    )

    if Path(test_path).exists() and Path(test_path).stat().st_size > 0:
        test_stats = _split_stats(test_path)
        print(
            f" | test={test_stats['total']} (biased={test_stats['biased']}, neutral={test_stats['neutral']})"
        )

def build_model_and_tokenizer(cfg: dict):
    print(f"⬇  Loading base model: {cfg['base_model']}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    model = AutoModelForTokenClassification.from_pretrained(
        cfg["base_model"],
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    return tokenizer, model


def build_training_args(cfg: dict) -> TrainingArguments:
    return TrainingArguments(
        output_dir=cfg["output_dir"],

        # ── Training schedule ──────────────────────────────
        num_train_epochs=cfg.get("epochs", 5),
        per_device_train_batch_size=cfg.get("batch_size", 16),
        per_device_eval_batch_size=cfg.get("batch_size", 16),
        gradient_accumulation_steps=cfg.get("grad_accum", 1),
        warmup_ratio=cfg.get("warmup_ratio", 0.1),
        learning_rate=cfg.get("lr", 2e-5),
        weight_decay=cfg.get("weight_decay", 0.01),
        lr_scheduler_type=cfg.get("lr_scheduler", "linear"),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),

        # ── Evaluation & checkpointing ─────────────────────
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,

        # ── Precision (use fp16 on NVIDIA, bf16 on A100/H100)
        fp16=cfg.get("fp16", False) and torch.cuda.is_available(),
        bf16=cfg.get("bf16", False),

        # ── Logging ────────────────────────────────────────
        logging_dir=f"{cfg['output_dir']}/logs",
        logging_steps=cfg.get("logging_steps", 50),
        report_to=cfg.get("report_to", "wandb"),

        # ── Misc ───────────────────────────────────────────
        seed=cfg.get("seed", 42),
        dataloader_num_workers=cfg.get("dataloader_workers", 2),
        push_to_hub=cfg.get("push_to_hub", False),
        hub_model_id=cfg.get("hub_model_id", None),
    )

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        weights = self.class_weights.to(logits.device)

        loss_fct = CrossEntropyLoss(
            weight=weights,
            ignore_index=-100
        )

        loss = loss_fct(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )

        return (loss, outputs) if return_outputs else loss
    
def main(config_path: str):
    cfg = load_config(config_path)
    print(f"\n{'='*55}")
    print(f"  JD Bias Detector — Training")
    print(f"  Base model : {cfg['base_model']}")
    print(f"  Output dir : {cfg['output_dir']}")
    print(f"  Epochs     : {cfg.get('epochs', 5)}")
    print(f"  Batch size : {cfg.get('batch_size', 16)}")
    print(f"  Device     : {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"{'='*55}\n")

    tokenizer, model = build_model_and_tokenizer(cfg)

    train_path = "data/annotated/train.jsonl"
    val_path = "data/annotated/val.jsonl"
    test_path = "data/annotated/test.jsonl"
    validate_data_splits(train_path, val_path, test_path)

    print("📂 Loading datasets ...")
    train_ds = BiasDataset(train_path, tokenizer,
                           max_length=cfg.get("max_length", 256))
    val_ds   = BiasDataset(val_path,   tokenizer,
                           max_length=cfg.get("max_length", 256))

    print(f"   Train: {len(train_ds)} samples")
    print(f"   Val:   {len(val_ds)} samples\n")

    data_collator = DataCollatorForTokenClassification(
        tokenizer, pad_to_multiple_of=8 if cfg.get("fp16") else None
    )

    training_args = build_training_args(cfg)

    callbacks = []
    if cfg.get("early_stopping_patience"):
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=cfg["early_stopping_patience"])
        )

    class_weights = compute_class_weights(
        train_path,
        LABEL2ID
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        class_weights=class_weights
    )

    print("🚀 Starting training ...\n")
    trainer.train()

    print(f"\n💾 Saving best model → {cfg['output_dir']}")
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])

    # ── Final eval on test set ─────────────────────────────
    if Path(test_path).exists() and Path(test_path).stat().st_size > 0:
        print("\n📊 Evaluating on test set ...")
        test_ds = BiasDataset(test_path, tokenizer,
                              max_length=cfg.get("max_length", 256))
        results = trainer.evaluate(test_ds)
        print("\n── Test Results ──────────────────────────")
        for k, v in results.items():
            print(f"  {k:<40} {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        print("──────────────────────────────────────────")

    if cfg.get("push_to_hub"):
        print(f"\n⬆  Pushing to HuggingFace Hub: {cfg.get('hub_model_id')}")
        trainer.push_to_hub()

    print("\n✅ Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/configs/deberta_base.yaml",
                        help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
