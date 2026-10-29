"""LoRA fine-tune Qwen2.5-0.5B as a 6-way email triage classifier (CPU). Needs requirements-train.txt.

Usage: python -m email_triage.train TRAIN.jsonl TEST.jsonl   (rows: from, subject, snippet, label)
This is the earlier 6-way scheme; the headline result uses triage_binary.py (binary important/not).
"""
from __future__ import annotations

import json
import os
import sys
import time

LABELS = ["action", "personal", "school", "finance", "notify", "promo"]
L2I = {lab: i for i, lab in enumerate(LABELS)}


def load(p: str) -> list[dict]:
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def text(r: dict) -> str:
    return f"From: {r['from']}\nSubject: {r['subject']}\n{r['snippet']}"[:600]


def main(train_path: str, test_path: str) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from sklearn.metrics import accuracy_score, classification_report
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    base = os.environ.get("BASE", "Qwen/Qwen2.5-0.5B")
    torch.manual_seed(0)
    train, test = load(train_path), load(test_path)
    tok = AutoTokenizer.from_pretrained(base)
    tok.pad_token = tok.pad_token or tok.eos_token

    def enc(rows):
        d = Dataset.from_dict({"text": [text(r) for r in rows], "label": [L2I[r["label"]] for r in rows]})
        return d.map(lambda b: tok(b["text"], truncation=True, max_length=160), batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=len(LABELS))
    model.config.pad_token_id = tok.pad_token_id
    model = get_peft_model(model, LoraConfig(task_type="SEQ_CLS", r=16, lora_alpha=32, lora_dropout=0.1,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    args = TrainingArguments("out", num_train_epochs=int(os.environ.get("EPOCHS", "3")), per_device_train_batch_size=8,
                             learning_rate=3e-4, logging_steps=20, save_strategy="no", report_to=[], use_cpu=True,
                             dataloader_num_workers=0)
    t0 = time.time()
    Trainer(model=model, args=args, train_dataset=enc(train), data_collator=DataCollatorWithPadding(tok)).train()
    print(f"train_seconds={time.time() - t0:.0f}")
    model.eval()
    preds = []
    t0 = time.time()
    with torch.no_grad():
        for r in test:
            x = tok(text(r), return_tensors="pt", truncation=True, max_length=160)
            preds.append(LABELS[int(model(**x).logits.argmax())])
    ms = (time.time() - t0) / len(test) * 1000
    gold = [r["label"] for r in test]
    print(f"test_n={len(test)} acc={accuracy_score(gold, preds):.3f} ms_per_msg={ms:.0f}")
    print(classification_report(gold, preds, labels=LABELS, zero_division=0))
    with open("preds.json", "w") as f:
        json.dump([{"id": r.get("id"), "gold": g, "pred": p} for r, g, p in zip(test, gold, preds)], f)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
