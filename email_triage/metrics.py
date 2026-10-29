"""Binary important/not metrics and result assembly (stdlib only, offline)."""
from __future__ import annotations

import json
import statistics

REQUIRED_FIELDS = ("tid", "name", "from", "subject", "snippet", "date", "labels", "split", "gold", "prod")


def load_rows(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def check_schema(rows: list[dict]) -> None:
    for r in rows:
        missing = [k for k in REQUIRED_FIELDS if k not in r]
        if missing:
            raise ValueError(f"row {r.get('tid')!r} missing {missing}")
        if r["split"] not in ("train", "test"):
            raise ValueError(f"bad split {r['split']!r}")


def prf(preds: list[dict]) -> dict:
    """preds: [{'gold': 0/1, 'pred': 0/1, 'ms': float}]. Returns acc/precision/recall/f1/confusion/latency."""
    tp = sum(1 for p in preds if p["gold"] and p["pred"])
    fp = sum(1 for p in preds if not p["gold"] and p["pred"])
    fn = sum(1 for p in preds if p["gold"] and not p["pred"])
    tn = sum(1 for p in preds if not p["gold"] and not p["pred"])
    n = len(preds)
    ms = sorted(p["ms"] for p in preds)
    return dict(
        acc=round((tp + tn) / n, 3),
        precision=round(tp / max(tp + fp, 1), 3),
        recall=round(tp / max(tp + fn, 1), 3),
        f1=round(2 * tp / max(2 * tp + fp + fn, 1), 3),
        tp=tp, fp=fp, fn=fn, tn=tn,
        p50_ms=round(statistics.median(ms)),
        p95_ms=round(ms[max(int(0.95 * n) - 1, 0)]),
    )


def production_preds(rows: list[dict]) -> list[dict]:
    """The incumbent rules' stored decisions on the test split, scored against gold."""
    return [{"gold": r["gold"], "pred": r["prod"], "ms": 0} for r in rows if r["split"] == "test"]


def build_results(rows: list[dict], model_preds: dict[str, dict], generated: str = "2026-10-27") -> dict:
    """model_preds: {'lora': {'base':..., 'preds': [...]}, ...} as written by triage_binary.py."""
    test_n = sum(1 for r in rows if r["split"] == "test")
    res = {
        "generated": generated,
        "test_n": test_n,
        "gold": "held-out emails hand-labeled by the author (single labeler); data not released for privacy",
        "train": "older emails labeled by an LLM from a written rubric (never used as a grader); data not released",
        "models": {},
    }
    res["models"]["production-rules"] = dict(prf(production_preds(rows)), latency="n/a (regex)", cost=0)
    for mode, d in model_preds.items():
        meta = {k: v for k, v in d.items() if k != "preds"}
        res["models"][mode] = {**meta, **prf(d["preds"]), "cost_usd": 0}
    return res
