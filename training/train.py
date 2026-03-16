"""
train.py — Fine-tune DeBERTa-v3-base for JD bias token classification

Usage (cloud GPU):
    python -m training.train --config training/configs/deberta_base.yaml

Colab/Kaggle one-liner:
    !python -m training.train --config training/configs/deberta_base.yaml
"""
import os
import argparse
import yaml
import torch
from pathlib import Path
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


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


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
        fp16=cfg.get("fp16", torch.cuda.is_available()),
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

    print("📂 Loading datasets ...")
    train_ds = BiasDataset("data/annotated/train.jsonl", tokenizer,
                           max_length=cfg.get("max_length", 256))
    val_ds   = BiasDataset("data/annotated/val.jsonl",   tokenizer,
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    print("🚀 Starting training ...\n")
    trainer.train()

    print(f"\n💾 Saving best model → {cfg['output_dir']}")
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])

    # ── Final eval on test set ─────────────────────────────
    test_path = "data/annotated/test.jsonl"
    if Path(test_path).exists():
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
