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
    d = json.loads((ROOT / "results/email-triage/lora-triage.json").read_text())
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
    preds = {"lora": {"base": "stub", "preds": [{"tid": x["tid"], "gold": x["gold"], "pred": x["gold"], "ms": 5}
                                                for x in test]}}
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


def test_bootstrap_ci_brackets_point_estimate():
    from email_triage.metrics import bootstrap_ci

    preds = [{"gold": i % 2, "pred": (i % 3 == 0) * 1, "ms": 1} for i in range(60)]
    lo, hi = bootstrap_ci(preds)
    assert lo <= prf(preds)["acc"] <= hi and 0 <= lo < hi <= 1
    assert bootstrap_ci(preds) == (lo, hi)  # seeded, so reproducible


def test_mcnemar_identical_and_lopsided():
    from email_triage.metrics import mcnemar_p

    a = [{"gold": 1, "pred": 1}] * 20
    assert mcnemar_p(a, a) == 1.0
    b = [{"gold": 1, "pred": 0}] * 20
    assert mcnemar_p(a, b) < 0.001


def test_mcnemar_exact_value():
    from email_triage.metrics import mcnemar_p

    # a right / b wrong on 1 email, the reverse on 6: exact two-sided binomial p = 2 * 7/128 = 0.125
    a = [{"gold": 1, "pred": 1}] + [{"gold": 1, "pred": 0}] * 6
    b = [{"gold": 1, "pred": 0}] + [{"gold": 1, "pred": 1}] * 6
    assert mcnemar_p(a, b) == 0.125


def test_mcnemar_rejects_unpaired_lists():
    from email_triage.metrics import mcnemar_p

    with pytest.raises(ValueError):
        mcnemar_p([{"gold": 1, "pred": 1}] * 3, [{"gold": 1, "pred": 1}] * 2)


def test_prf_latency_nearest_rank_and_empty():
    preds = [{"gold": 1, "pred": 1, "ms": float(i)} for i in range(1, 151)]
    m = prf(preds)
    assert m["p95_ms"] == 143.0  # nearest rank: ceil(0.95 * 150) = 143rd value
    assert prf([{"gold": 0, "pred": 0, "ms": 0.44}])["p50_ms"] == 0.4
    with pytest.raises(ValueError):
        prf([])


def _lora_preds(test, tids=None, gold_flip=None):
    tids = tids or [x["tid"] for x in test]
    by_tid = {x["tid"]: x for x in test}
    return [{"tid": t, "gold": by_tid.get(t, test[0])["gold"] ^ (t == gold_flip), "pred": 1, "ms": 1} for t in tids]


def test_build_results_aligns_shuffled_tids():
    r = rows()
    test = [x for x in r if x["split"] == "test"]
    shuffled = [x["tid"] for x in reversed(test)]
    res = build_results(r, {"lora": {"preds": _lora_preds(test, shuffled)},
                            "tfidf": {"preds": _lora_preds(test)}})
    assert res["mcnemar_p_vs_lora"]["tfidf"] == 1.0


@pytest.mark.parametrize("case", ["missing", "unknown", "duplicate", "gold"])
def test_build_results_rejects_bad_predictions(case):
    r = rows()
    test = [x for x in r if x["split"] == "test"]
    tids = [x["tid"] for x in test]
    preds = {
        "missing": _lora_preds(test, tids[:-1]),
        "unknown": _lora_preds(test, tids[:-1] + ["not-a-test-tid"]),
        "duplicate": _lora_preds(test, tids[:-1] + tids[:1]),
        "gold": _lora_preds(test, gold_flip=tids[3]),
    }[case]
    with pytest.raises(ValueError):
        build_results(r, {"lora": {"preds": preds}})


def test_tfidf_baseline_runs_on_fixture(tmp_path):
    pytest.importorskip("sklearn")
    from email_triage import baseline

    out = baseline.run(str(FX / "emails_synth.jsonl"), str(FX / "train_labels_synth.tsv"), str(tmp_path))
    assert len(out["preds"]) == 10 and out["train_n"] == 10
    res = make_results.main(str(FX / "emails_synth.jsonl"), str(tmp_path), str(tmp_path / "o.json"))
    m = res["models"]["tfidf"]
    assert m["tp"] + m["fp"] + m["fn"] + m["tn"] == 10
    assert 0.0 <= m["acc"] <= 1.0


def test_published_results_have_baseline_and_ci():
    d = json.loads((ROOT / "results/email-triage/lora-triage.json").read_text())
    assert {"production-rules", "lora", "zeroshot", "tfidf"} <= set(d["models"])
    for m in d["models"].values():
        lo, hi = m["acc_ci"]
        assert lo <= m["acc"] <= hi


def test_chart_renders_svg():
    from email_triage.chart import render

    d = json.loads((ROOT / "results/email-triage/lora-triage.json").read_text())
    svg = render(d)
    assert svg.startswith("<svg") and "LoRA 0.5B" in svg and "TF-IDF" in svg
    # the committed chart is exactly what render() produces from the committed results
    assert svg + "\n" == (ROOT / "docs/email-triage.svg").read_text()


def test_best_threshold_maximizes_train_accuracy():
    from email_triage.triage_binary import best_threshold

    # every score is positive, so threshold 0 calls everything important; the best cut separates 2.0 from 5.0
    scores, labels = [1.0, 2.0, 5.0, 6.0], [0, 0, 1, 1]
    t = best_threshold(scores, labels)
    assert 2.0 < t < 5.0
    assert [int(s > t) for s in scores] == labels
    assert best_threshold([-1.0, 1.0], [0, 1]) == 0.0  # ties prefer the uncalibrated threshold


def test_label_logprob_sums_every_label_token():
    torch = pytest.importorskip("torch")
    from email_triage.triage_binary import label_logprob

    class Uniform:  # every next token equally likely over a 4-token vocabulary
        def __call__(self, input_ids):
            return type("Out", (), {"logits": torch.zeros(1, input_ids.shape[1], 4)})()

    lp = label_logprob(Uniform(), torch.tensor([0, 1]), [2, 3, 2])
    assert lp == pytest.approx(3 * torch.log(torch.tensor(0.25)).item())
