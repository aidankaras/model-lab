"""Classic baseline: TF-IDF (word 1-2 grams + char 3-5 grams) + logistic regression.

Same inputs and output format as triage_binary.py, so make_results picks it up as `tfidf`.
Needs scikit-learn (requirements-train.txt).

Usage: python -m email_triage.baseline DATA.jsonl TRAIN_LABELS.tsv OUT_DIR
"""
from __future__ import annotations

import json
import os
import sys
import time

from email_triage.triage_binary import email_text, load_train_labels, split_rows


def run(data: str, labels_path: str, out_dir: str, seed: int = 0) -> dict:
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    with open(data) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    train, test = split_rows(rows, load_train_labels(labels_path))
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    xtr_text = [email_text(r) for r in train]
    t0 = time.time()
    xtr = hstack([word.fit_transform(xtr_text), char.fit_transform(xtr_text)]).tocsr()
    clf = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced", random_state=seed)
    clf.fit(xtr, [r["y"] for r in train])
    train_s = time.time() - t0
    out = []
    for r in test:
        t = time.time()
        x = hstack([word.transform([email_text(r)]), char.transform([email_text(r)])]).tocsr()
        p = int(clf.predict(x)[0])
        out.append({"tid": r["tid"], "gold": r["y"], "pred": p, "ms": (time.time() - t) * 1000})
    meta = {"base": "tfidf-word1-2+char3-5 + logistic-regression", "train_n": len(train),
            "train_seconds": round(train_s, 2)}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "preds-tfidf.json"), "w") as f:
        json.dump({"mode": "tfidf", **meta, "preds": out}, f)
    acc = sum(o["gold"] == o["pred"] for o in out) / len(out)
    print("tfidf", meta, "acc", round(acc, 3))
    return {"mode": "tfidf", **meta, "preds": out}


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    run(*sys.argv[1:])
