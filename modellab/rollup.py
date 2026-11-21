"""Recompute the README's model-comparison numbers from the committed result JSONs.

Usage: python -m modellab.rollup [RESULTS_DIR]   (default: results)

Reads e08b-effort-ladder.json, e08b-three-graders.json and grader-robustness.json and prints, per model:
pass counts and cost per passed task under the single grader and under the majority of three graders,
the grader spread on the graders that scored every cell, and which cells break cross-family grading
(checked with scoring.grader_conflicts). Cost is candidate spend only; grader spend is reported separately.
"""
from __future__ import annotations

import json
import os
import sys

from modellab.scoring import cost_per_pass, grader_conflicts

LEVELS = ("low", "medium", "high")


def _load(results_dir: str, name: str) -> dict:
    with open(os.path.join(results_dir, name)) as f:
        return json.load(f)


def _short(name: str) -> str:
    """'deepseek-v4.1-flash' / 'deepseek_candidate' / 'deepseek-low' -> 'deepseek'."""
    return name.replace("_", "-").split("-")[0]


def effort_ladder(results_dir: str) -> dict[str, dict]:
    """Per model: runs, candidate cost, and passes / cost per pass under luna alone and under majority-of-3."""
    ladder = _load(results_dir, "e08b-effort-ladder.json")
    graded = _load(results_dir, "e08b-three-graders.json")
    ids = {_short(m.split("/")[-1]): m for m in ladder["models"]}
    out = {}
    for key, levels in ladder["pass_counts_out_of_12_runs_per_level"].items():
        m = _short(key)
        cost = next(v for k, v in ladder["cost"].items() if _short(k) == m)
        cells = {lv: graded["cells"][f"{m}-{lv}"] for lv in LEVELS}
        luna = sum(levels[lv] for lv in LEVELS)
        if luna != cost["passed"] or any(cells[lv]["luna"] != levels[lv] for lv in LEVELS):
            raise ValueError(f"{m}: luna pass counts disagree between the effort-ladder and three-grader files")
        majority = sum(c["majority"] for c in cells.values())
        conflicts = [g for g, gid in graded["graders"].items() if grader_conflicts(gid, [ids[m]])]
        out[m] = {
            "model": ids[m],
            "runs": 12 * len(LEVELS),
            "cost_usd": cost["usd"],
            "passed_luna": luna,
            "cost_per_pass_luna": cost_per_pass(cost["usd"], luna),
            "passed_majority": majority,
            "cost_per_pass_majority": cost_per_pass(cost["usd"], majority),
            "majority_conflicting_graders": conflicts,
        }
    return out


def grader_spread(results_dir: str) -> dict[str, dict]:
    """Per model: min/max spread (max - min passes) over the graders that scored every cell, and over all graders."""
    d = _load(results_dir, "grader-robustness.json")
    graders = list(d["graders"])
    common = [g for g in graders if all(r[g] is not None for r in d["rows"])]
    out: dict[str, dict] = {}
    for r in d["rows"]:
        m = _short(r["cell"])
        on_common = [r[g] for g in common]
        on_all = [r[g] for g in graders if r[g] is not None]
        s = out.setdefault(m, {"common_graders": common, "common": [], "all": [], "n_graders": set()})
        s["common"].append(max(on_common) - min(on_common))
        s["all"].append(max(on_all) - min(on_all))
        s["n_graders"].add(len(on_all))
    return {m: {"common_graders": s["common_graders"], "spread_common": (min(s["common"]), max(s["common"])),
                "spread_all": (min(s["all"]), max(s["all"])), "n_graders_all": sorted(s["n_graders"])}
            for m, s in out.items()}


def contaminated_cells(results_dir: str) -> list[tuple[str, str, str]]:
    """(file, grader, cell) for every graded cell whose grader shares a family with the candidate."""
    ids = {_short(m.split("/")[-1]): m for m in _load(results_dir, "e08b-effort-ladder.json")["models"]}
    found = []
    for name, cells in (("e08b-three-graders.json", None), ("grader-robustness.json", "rows")):
        d = _load(results_dir, name)
        rows = d[cells] if cells else [{"cell": k, **v} for k, v in d["cells"].items()]
        for r in rows:
            for g, gid in d["graders"].items():
                if r.get(g) is not None and grader_conflicts(gid, [ids[_short(r["cell"])]]):
                    found.append((name, g, r["cell"]))
    return found


def recorded_spend(results_dir: str) -> float:
    """Sum of cost_usd_recorded across result files (grader passes without a recorded cost are not included)."""
    total = 0.0
    for name in sorted(os.listdir(results_dir)):
        if name.endswith(".json"):
            total += _load(results_dir, name).get("cost_usd_recorded") or 0.0
    return round(total, 4)


def _usd(x: float | None) -> str:
    return "undefined" if x is None else f"${x:.4f}"


def main(results_dir: str = "results") -> dict:
    ladder, spread = effort_ladder(results_dir), grader_spread(results_dir)
    print(f"Effort ladder: {sum(v['runs'] for v in ladder.values())} runs, {len(ladder)} models\n")
    print("| Model | Passed (luna) | Cost/pass (luna) | Passed (majority of 3) | Cost/pass (majority) | Spread, common graders |")
    print("|---|---|---|---|---|---|")
    for m, v in ladder.items():
        flag = f" (contaminated: {', '.join(v['majority_conflicting_graders'])})" if v["majority_conflicting_graders"] else ""
        lo, hi = spread[m]["spread_common"]
        print(f"| {v['model']} | {v['passed_luna']}/{v['runs']} | {_usd(v['cost_per_pass_luna'])} | "
              f"{v['passed_majority']}/{v['runs']}{flag} | {_usd(v['cost_per_pass_majority'])} | {lo}-{hi} |")
    print("\nCross-family violations:")
    for f, g, c in contaminated_cells(results_dir):
        print(f"- {f}: grader {g} on {c}")
    print(f"\nRecorded spend across result files: ${recorded_spend(results_dir):.2f}")
    return {"ladder": ladder, "spread": spread}


if __name__ == "__main__":
    main(*sys.argv[1:2])
