"""Binary important/not metrics and result assembly (stdlib only, offline)."""
from __future__ import annotations

import json
import math
import statistics

REQUIRED_FIELDS = ("tid", "name", "from", "subject", "snippet", "date", "labels", "split", "gold", "prod")


def load_rows(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def check_schema(rows: list[dict]) -> None:
    for r in rows:
        # gold labels exist only for the held-out test split; train labels live in a separate TSV
        missing = [k for k in REQUIRED_FIELDS if k not in r and not (k == "gold" and r.get("split") == "train")]
        if missing:
            raise ValueError(f"row {r.get('tid')!r} missing {missing}")
        if r["split"] not in ("train", "test"):
            raise ValueError(f"bad split {r['split']!r}")


def prf(preds: list[dict]) -> dict:
    """preds: [{'gold': 0/1, 'pred': 0/1, 'ms': float}]. Returns acc/precision/recall/f1/confusion/latency.

    Latency percentiles use the nearest-rank rule and keep one decimal, so sub-millisecond systems
    are not rounded to a whole number.
    """
    if not preds:
        raise ValueError("prf() needs at least one prediction")
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
        p50_ms=round(statistics.median(ms), 1),
        p95_ms=round(ms[math.ceil(0.95 * n) - 1], 1),
    )


def bootstrap_ci(preds: list[dict], metric: str = "acc", n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap CI for a prf() metric, resampling emails with replacement.

    With the default 2000 resamples this trims 50 values from each end (indices 50 and 1949).
    """
    import random

    rng = random.Random(seed)
    n = len(preds)
    vals = sorted(prf([preds[rng.randrange(n)] for _ in range(n)])[metric] for _ in range(n_boot))
    return round(vals[int(0.025 * n_boot)], 3), round(vals[int(0.975 * n_boot) - 1], 3)


def mcnemar_p(a: list[dict], b: list[dict]) -> float:
    """Exact two-sided McNemar p-value for paired predictions a vs b on the same emails (same order)."""
    from math import comb

    if len(a) != len(b):
        raise ValueError(f"mcnemar_p needs paired predictions; got {len(a)} vs {len(b)}")
    b01 = sum(1 for x, y in zip(a, b) if (x["pred"] == x["gold"]) and (y["pred"] != y["gold"]))
    b10 = sum(1 for x, y in zip(a, b) if (x["pred"] != x["gold"]) and (y["pred"] == y["gold"]))
    n, k = b01 + b10, min(b01, b10)
    if n == 0:
        return 1.0
    return round(min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n), 4)


def production_preds(rows: list[dict]) -> list[dict]:
    """The incumbent rules' stored decisions on the test split, scored against gold."""
    return [{"gold": r["gold"], "pred": r["prod"], "ms": 0} for r in rows if r["split"] == "test"]


def align_preds(mode: str, preds: list[dict], test_rows: list[dict], order: dict[str, int]) -> list[dict]:
    """Put preds in test-row order. Raise if any test email is missing, unknown, duplicated or has a different gold."""
    tids = [p.get("tid") for p in preds]
    if len(tids) != len(set(tids)):
        raise ValueError(f"{mode}: duplicate tids in predictions")
    missing, extra = set(order) - set(tids), set(tids) - set(order)
    if missing or extra:
        raise ValueError(f"{mode}: predictions do not match the test split "
                         f"(missing {sorted(missing)[:5]}, unknown {sorted(map(str, extra))[:5]})")
    p = sorted(preds, key=lambda x: order[x["tid"]])
    bad_gold = [x["tid"] for x, r in zip(p, test_rows) if x["gold"] != r["gold"]]
    if bad_gold:
        raise ValueError(f"{mode}: gold label differs from the dataset for {bad_gold[:5]}")
    return p


def build_results(rows: list[dict], model_preds: dict[str, dict]) -> dict:
    """model_preds: {'lora': {'base':..., 'preds': [...]}, ...} as written by triage_binary.py / baseline.py."""
    test_n = sum(1 for r in rows if r["split"] == "test")
    res = {
        "test_n": test_n,
        "ci": "95% percentile bootstrap over emails, 2000 resamples, seed 0",
        "gold": "held-out emails hand-labeled by the author (single labeler); data not released for privacy",
        "train": "older emails labeled by an LLM from a written rubric (never used as a grader); data not released",
        "models": {},
    }
    prod = production_preds(rows)
    res["models"]["production-rules"] = dict(prf(prod), acc_ci=bootstrap_ci(prod), f1_ci=bootstrap_ci(prod, "f1"),
                                             latency="n/a (regex)", cost_usd=0)
    test_rows = [r for r in rows if r["split"] == "test"]
    order = {r["tid"]: i for i, r in enumerate(test_rows)}
    aligned = {}
    for mode, d in model_preds.items():
        meta = {k: v for k, v in d.items() if k not in ("preds", "mode")}
        p = align_preds(mode, d["preds"], test_rows, order)
        aligned[mode] = p
        res["models"][mode] = {**meta, **prf(p), "acc_ci": bootstrap_ci(p), "f1_ci": bootstrap_ci(p, "f1"),
                               "cost_usd": 0}
    if "lora" in aligned:
        res["mcnemar_p_vs_lora"] = {m: mcnemar_p(aligned["lora"], p) for m, p in aligned.items() if m != "lora"}
        res["mcnemar_p_vs_lora"]["production-rules"] = mcnemar_p(aligned["lora"], prod)
    return res
