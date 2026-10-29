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
