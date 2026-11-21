import pytest

from modellab.scoring import assert_grader_separate, cost_per_pass, family, grader_conflicts, summarize, task_score


def test_cost_per_pass_arithmetic():
    assert cost_per_pass(0.0157, 17) == pytest.approx(0.000924, rel=1e-2)
    assert cost_per_pass(1.0, 4) == 0.25


def test_cost_per_pass_undefined_when_none_pass():
    assert cost_per_pass(1.0, 0) is None


def test_fabrication_is_automatic_fail():
    assert task_score(True, 0)
    assert not task_score(True, 1)
    assert not task_score(False, 0)


def test_summarize():
    tasks = [{"pass": True}, {"pass": True, "fabrications": 2}, {"pass": False}, {"pass": True}]
    s = summarize(tasks, 0.3)
    assert s["passed"] == 2 and s["n"] == 4 and s["cost_per_pass"] == pytest.approx(0.15)


def test_family_mapping():
    assert family("z-ai/glm-5.3-flash") == "zhipu"
    assert family("deepseek/deepseek-v4-pro") == "deepseek"
    assert family("moonshotai/kimi-k2.6") == "moonshot"


def test_grader_must_differ_from_candidates():
    assert grader_conflicts("openai/gpt-6-luna", ["moonshotai/kimi-k2.6", "z-ai/glm-5.3-flash"]) == []
    assert_grader_separate("openai/gpt-6-luna", ["z-ai/glm-5.3-flash"])
    with pytest.raises(ValueError):
        assert_grader_separate("deepseek/deepseek-v4-pro", ["deepseek/deepseek-v4-flash"])


def test_family_openai_o_series_and_unknown_vendor():
    assert family("openai/o3") == "openai" and family("o4-mini") == "openai"
    assert family("acme/foo-1") == "acme"


def test_rollup_reproduces_readme_numbers():
    from pathlib import Path

    from modellab import rollup

    res = str(Path(__file__).parent.parent / "results")
    ladder = rollup.effort_ladder(res)
    assert sum(v["runs"] for v in ladder.values()) == 108
    got = {m: (v["passed_luna"], round(v["cost_per_pass_luna"], 4), v["passed_majority"],
               round(v["cost_per_pass_majority"], 4)) for m, v in ladder.items()}
    assert got == {"kimi": (17, 0.0129, 29, 0.0075), "deepseek": (14, 0.0015, 16, 0.0013),
                   "glm": (17, 0.0009, 17, 0.0009)}
    assert ladder["deepseek"]["majority_conflicting_graders"] == ["v4pro"]
    assert ladder["glm"]["majority_conflicting_graders"] == []

    spread = rollup.grader_spread(res)
    assert {m: s["spread_common"] for m, s in spread.items()} == {"kimi": (5, 7), "deepseek": (6, 7), "glm": (2, 4)}
    assert spread["kimi"]["spread_all"] == (6, 8) and spread["deepseek"]["spread_all"] == (8, 9)

    bad = rollup.contaminated_cells(res)
    assert bad and all(cell.startswith("deepseek-") and g == "v4pro" for _, g, cell in bad)
    assert rollup.recorded_spend(res) == pytest.approx(0.61, abs=0.01)


def test_rollup_cli_prints_table(capsys):
    from pathlib import Path

    from modellab import rollup

    rollup.main(str(Path(__file__).parent.parent / "results"))
    out = capsys.readouterr().out
    assert "108 runs" in out and "| moonshotai/kimi-k2.6 | 17/36 | $0.0129 | 29/36 | $0.0075 | 5-7 |" in out
