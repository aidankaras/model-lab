"""Scoring helpers: cost per passed task and grader/candidate family separation.

Pure stdlib so it runs offline.
"""
from __future__ import annotations

# Map a model id (or loose name) to a vendor family. Order matters: first match wins.
_FAMILY_HINTS = (
    ("moonshot", "moonshot"), ("kimi", "moonshot"),
    ("deepseek", "deepseek"),
    ("z-ai", "zhipu"), ("glm", "zhipu"),
    ("openai", "openai"), ("gpt", "openai"),
    ("google", "google"), ("gemini", "google"),
    ("anthropic", "anthropic"), ("claude", "anthropic"), ("sonnet", "anthropic"),
    ("stepfun", "stepfun"), ("step-", "stepfun"),
    ("qwen", "alibaba"),
)


def family(model_id: str) -> str:
    """Vendor family of a model id, e.g. 'z-ai/glm-5.3-flash' -> 'zhipu'."""
    m = model_id.lower()
    for hint, fam in _FAMILY_HINTS:
        if hint in m:
            return fam
    return m.split("/")[0]


def cost_per_pass(cost_usd: float, passed: int) -> float | None:
    """Cost per *passed* task. None when nothing passed (undefined, not zero)."""
    if passed <= 0:
        return None
    return cost_usd / passed


def grader_conflicts(grader: str, candidates: list[str]) -> list[str]:
    """Candidates that share a family with the grader (empty list = clean)."""
    gf = family(grader)
    return [c for c in candidates if family(c) == gf]


def assert_grader_separate(grader: str, candidates: list[str]) -> None:
    bad = grader_conflicts(grader, candidates)
    if bad:
        raise ValueError(f"grader {grader!r} shares a family with candidate(s): {bad}")


def task_score(passed: bool, fabrications: int = 0) -> bool:
    """A task passes only if the grader passed it AND nothing was fabricated."""
    return bool(passed) and fabrications == 0


def summarize(tasks: list[dict], cost_usd: float) -> dict:
    """Roll per-task records ({'pass': bool, 'fabrications': int}) into the headline numbers."""
    passed = sum(task_score(t.get("pass", False), t.get("fabrications", 0)) for t in tasks)
    return {"n": len(tasks), "passed": passed, "cost_usd": cost_usd, "cost_per_pass": cost_per_pass(cost_usd, passed)}
