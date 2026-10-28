"""Assemble results JSON from a triage dataset plus preds-<mode>.json files.

Usage: python -m email_triage.make_results DATA.jsonl PREDS_DIR OUT.json
"""
from __future__ import annotations

import json
import sys

from email_triage.metrics import build_results, check_schema, load_rows


def main(data: str, preds_dir: str, out: str) -> dict:
    rows = load_rows(data)
    check_schema(rows)
    model_preds = {}
    for mode in ("lora", "zeroshot", "tfidf"):
        try:
            with open(f"{preds_dir}/preds-{mode}.json") as f:
                model_preds[mode] = json.load(f)
        except FileNotFoundError:
            continue
    res = build_results(rows, model_preds)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    for k, v in res["models"].items():
        print(k, {x: v[x] for x in ("acc", "precision", "recall", "f1", "p50_ms", "p95_ms") if x in v})
    return res


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
