import json
from pathlib import Path

import pytest

from email_triage import make_results
from email_triage.metrics import REQUIRED_FIELDS, build_results, check_schema, load_rows, prf, production_preds
from email_triage.triage_binary import email_text, load_train_labels, split_rows

FX = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).parent.parent


def rows():
    return load_rows(str(FX / "emails_synth.jsonl"))


def test_fixture_schema_and_size():
    r = rows()
    check_schema(r)
    assert len(r) == 20
    assert {x["split"] for x in r} == {"train", "test"}
    assert set(r[0]) == set(REQUIRED_FIELDS)


def test_schema_rejects_missing_field():
    bad = [{"tid": "x"}]
    with pytest.raises(ValueError):
        check_schema(bad)


def test_prf_hand_computed():
    preds = [{"gold": 1, "pred": 1, "ms": 10}, {"gold": 1, "pred": 0, "ms": 20},
             {"gold": 0, "pred": 1, "ms": 30}, {"gold": 0, "pred": 0, "ms": 40}]
    m = prf(preds)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 1)
    assert m["acc"] == 0.5 and m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5


def test_published_numbers_are_self_consistent():
    d = json.loads((ROOT / "results/email-triage/results-2026-10-27-lora-triage.json").read_text())
    assert d["test_n"] == 150
    for name, m in d["models"].items():
        assert m["tp"] + m["fp"] + m["fn"] + m["tn"] == 150, name
        assert m["acc"] == pytest.approx((m["tp"] + m["tn"]) / 150, abs=1e-3)
        assert m["f1"] == pytest.approx(2 * m["tp"] / (2 * m["tp"] + m["fp"] + m["fn"]), abs=1e-3)


def test_production_rules_on_fixture():
    m = prf(production_preds(rows()))
    assert m["tp"] + m["fn"] == 5 and m["tp"] == 1


def test_build_results_and_cli(tmp_path):
    r = rows()
    test = [x for x in r if x["split"] == "test"]
    preds = {"lora": {"base": "stub", "preds": [{"gold": x["gold"], "pred": x["gold"], "ms": 5} for x in test]}}
    res = build_results(r, preds)
    assert res["models"]["lora"]["acc"] == 1.0 and res["test_n"] == 10
    (tmp_path / "preds-lora.json").write_text(json.dumps(preds["lora"]))
    out = tmp_path / "o.json"
    make_results.main(str(FX / "emails_synth.jsonl"), str(tmp_path), str(out))
    assert json.loads(out.read_text())["models"]["lora"]["f1"] == 1.0


def test_split_and_text():
    r = rows()
    labels = load_train_labels(str(FX / "train_labels_synth.tsv"))
    train, test = split_rows(r, labels)
    assert len(train) == 10 and len(test) == 10
    assert email_text(r[0]).startswith("From: Registrar Office")
