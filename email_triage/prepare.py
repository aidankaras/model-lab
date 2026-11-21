"""Turn pull.py output into the dataset schema the rest of the pipeline reads (see metrics.REQUIRED_FIELDS).

Usage: python -m email_triage.prepare RAW.jsonl OUT.jsonl [--test-n 150] [--prod-label LABEL] [--gold GOLD.tsv]

- `id` becomes `tid`; the From header "Name <addr>" is split into `name` and `from`.
- Rows are sorted by their Date header. The newest --test-n rows are the test split and the rest are train,
  so systems are always tested on mail that arrived after everything they trained on.
- `prod` is the incumbent filter's decision: 1 if the message carries --prod-label (the label your existing
  filter applies to mail it flags as important), else 0. In the original study `prod` held the stored
  decisions of the author's own regex mail filter; those rules are part of a private mail setup and are not
  in this repo.
- `gold` comes from GOLD.tsv (`tid<TAB>0|1`, your hand labels) for test rows. Run once without --gold to get
  OUT.jsonl plus OUT.test-tids.txt listing the emails to label, then again with --gold.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from email.utils import parseaddr, parsedate_to_datetime

OLDEST = datetime.min.replace(tzinfo=timezone.utc)


def parse_date(value: str) -> datetime | None:
    try:
        d = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def prepare(raw: list[dict], test_n: int, prod_label: str | None = None,
            gold: dict[str, int] | None = None) -> list[dict]:
    if not 0 < test_n < len(raw):
        raise ValueError(f"--test-n must be between 1 and {len(raw) - 1}, got {test_n}")
    dated = sorted(raw, key=lambda r: (parse_date(r.get("date", "")) or OLDEST, r["id"]))
    rows = []
    for i, r in enumerate(dated):
        name, addr = parseaddr(r.get("from", ""))
        split = "test" if i >= len(dated) - test_n else "train"
        row = {"tid": r["id"], "name": name or addr, "from": addr, "subject": r.get("subject", ""),
               "snippet": r.get("snippet", ""), "date": r.get("date", ""), "labels": r.get("labels", []),
               "split": split, "prod": int(bool(prod_label) and prod_label in r.get("labels", []))}
        if split == "test" and gold is not None:
            if r["id"] not in gold:
                raise ValueError(f"no gold label for test email {r['id']!r}")
            row["gold"] = gold[r["id"]]
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> list[dict]:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument("--test-n", type=int, default=150)
    ap.add_argument("--prod-label", default=None)
    ap.add_argument("--gold", default=None)
    a = ap.parse_args(argv)
    with open(a.raw) as f:
        raw = [json.loads(line) for line in f if line.strip()]
    gold = None
    if a.gold:
        with open(a.gold) as f:
            gold = {ln.split("\t")[0]: int(ln.split("\t")[1]) for ln in f if ln.strip()}
    rows = prepare(raw, a.test_n, a.prod_label, gold)
    undated = sum(parse_date(r.get("date", "")) is None for r in raw)
    if undated:
        print(f"warning: {undated} rows had an unparseable Date and were placed in train (oldest)")
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    if gold is None:
        with open(a.out.removesuffix(".jsonl") + ".test-tids.txt", "w") as f:
            f.write("".join(r["tid"] + "\n" for r in rows if r["split"] == "test"))
        print("label the test emails listed in the .test-tids.txt file, then re-run with --gold")
    return rows


if __name__ == "__main__":
    main()
