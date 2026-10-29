"""Binary important/not email triage (time-blocked split). Needs torch/transformers/peft (requirements-train.txt).

Usage: python -m email_triage.triage_binary {lora|zeroshot} DATA.jsonl TRAIN_LABELS.tsv OUT_DIR

DATA.jsonl rows follow the schema in email_triage/metrics.py (tid,name,from,subject,snippet,date,labels,split,gold,prod).
TRAIN_LABELS.tsv: `tid<TAB>0|1` labels for train rows (LLM-made in the original run; the gold test labels are human).
Writes OUT_DIR/preds-<mode>.json with per-email latency.
"""
from __future__ import annotations

import json
import os
import sys
import time


def email_text(r: dict) -> str:
    return f"From: {r['name']} <{r['from']}>\nSubject: {r['subject']}\n{r['snippet']}"[:600]


def load_train_labels(path: str) -> dict[str, int]:
    with open(path) as f:
        return {ln.split("\t")[0]: int(ln.split("\t")[1]) for ln in f if ln.strip()}


def split_rows(rows: list[dict], labels: dict[str, int]) -> tuple[list[dict], list[dict]]:
    train = [dict(r, y=labels[r["tid"]]) for r in rows if r["split"] == "train"]
    test = [dict(r, y=r["gold"]) for r in rows if r["split"] == "test"]
    return train, test


SYSTEM_PROMPT = (
    "You triage email for a university student who also has a part-time job and runs side projects. "
    "Answer with exactly one word: IMPORTANT (needs their attention: real people, school/job matters, money, "
    "orders, account security, services they pay for) or SKIP (marketing, newsletters, social nudges, automated reports)."
)


def run(mode: str, data: str, labels_path: str, out_dir: str) -> None:
    import torch  # heavy deps are imported lazily so the module imports without them
    from transformers import AutoTokenizer

    with open(data) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    train, test = split_rows(rows, load_train_labels(labels_path))
    torch.manual_seed(int(os.environ.get("SEED", "0")))
    torch.set_num_threads(int(os.environ.get("THREADS", "6")))
    out = []
    if mode == "lora":
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments

        base = os.environ.get("BASE", "Qwen/Qwen2.5-0.5B")
        tok = AutoTokenizer.from_pretrained(base)
        tok.pad_token = tok.pad_token or tok.eos_token
        m = AutoModelForSequenceClassification.from_pretrained(base, num_labels=2)
        m.config.pad_token_id = tok.pad_token_id
        m = get_peft_model(m, LoraConfig(task_type="SEQ_CLS", r=16, lora_alpha=32, lora_dropout=0.1,
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
        ds = Dataset.from_dict({"text": [email_text(r) for r in train], "label": [r["y"] for r in train]}).map(
            lambda b: tok(b["text"], truncation=True, max_length=160), batched=True)
        t0 = time.time()
        Trainer(model=m, args=TrainingArguments("out-bin", num_train_epochs=int(os.environ.get("EPOCHS", "4")),
                per_device_train_batch_size=8, learning_rate=3e-4, logging_steps=10, save_strategy="no",
                report_to=[], use_cpu=True), train_dataset=ds, data_collator=DataCollatorWithPadding(tok)).train()
        train_s = time.time() - t0
        m.eval()
        for r in test:
            t = time.time()
            with torch.no_grad():
                p = int(m(**tok(email_text(r), return_tensors="pt", truncation=True, max_length=160)).logits.argmax())
            out.append({"tid": r["tid"], "gold": r["y"], "pred": p, "ms": (time.time() - t) * 1000})
        meta = {"base": base, "train_n": len(train), "train_seconds": round(train_s)}
    else:
        from transformers import AutoModelForCausalLM

        base = os.environ.get("BASE", "Qwen/Qwen2.5-1.5B-Instruct")
        tok = AutoTokenizer.from_pretrained(base)
        m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32).eval()
        yes, no = tok.encode("IMPORTANT")[0], tok.encode("SKIP")[0]
        for r in test:
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": email_text(r)}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
            t = time.time()
            with torch.no_grad():
                lg = m(**ids).logits[0, -1]
            out.append({"tid": r["tid"], "gold": r["y"], "pred": int(lg[yes] > lg[no]), "ms": (time.time() - t) * 1000})
        meta = {"base": base}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"preds-{mode}.json"), "w") as f:
        json.dump({"mode": mode, **meta, "preds": out}, f)
    acc = sum(o["gold"] == o["pred"] for o in out) / len(out)
    print(mode, meta, "acc", round(acc, 3))


if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[1] not in ("lora", "zeroshot"):
        sys.exit(__doc__)
    run(*sys.argv[1:])
