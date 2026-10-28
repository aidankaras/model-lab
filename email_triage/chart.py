"""Render the email-triage accuracy/F1 chart as a dependency-free SVG.

Usage: python -m email_triage.chart results/email-triage/lora-triage.json docs/email-triage.svg
"""
from __future__ import annotations

import json
import sys

LABELS = {
    "production-rules": "Regex rules",
    "zeroshot": "Zero-shot 1.5B",
    "tfidf": "TF-IDF + LogReg",
    "lora": "LoRA 0.5B",
}
COLORS = {"acc": "#4C78A8", "f1": "#F58518"}


def render(results: dict) -> str:
    models = [m for m in LABELS if m in results["models"]]
    w, h, left, top, plot_h = 640, 320, 60, 40, 220
    group_w = (w - left - 20) / len(models)
    bar_w = group_w * 0.3

    def y(v: float) -> float:
        return top + plot_h * (1 - v)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" font-family="sans-serif" font-size="12">',
        f'<rect width="{w}" height="{h}" fill="white"/>',
        f'<text x="{w / 2}" y="20" text-anchor="middle" font-size="14" font-weight="bold">'
        f'Email triage on {results["test_n"]} held-out emails (95% bootstrap CI)</text>',
    ]
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        parts.append(f'<line x1="{left}" x2="{w - 20}" y1="{y(t)}" y2="{y(t)}" stroke="#ddd"/>')
        parts.append(f'<text x="{left - 6}" y="{y(t) + 4}" text-anchor="end">{t:.2f}</text>')
    for i, m in enumerate(models):
        d = results["models"][m]
        gx = left + i * group_w + group_w * 0.2
        for j, metric in enumerate(("acc", "f1")):
            v, (lo, hi) = d[metric], d[f"{metric}_ci"]
            x = gx + j * bar_w
            parts.append(f'<rect x="{x:.1f}" y="{y(v):.1f}" width="{bar_w:.1f}" height="{plot_h * v:.1f}" '
                         f'fill="{COLORS[metric]}"/>')
            cx = x + bar_w / 2
            parts.append(f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{y(hi):.1f}" y2="{y(lo):.1f}" stroke="black"/>')
            parts.append(f'<text x="{cx:.1f}" y="{y(hi) - 4:.1f}" text-anchor="middle" font-size="10">{v:.2f}</text>')
        parts.append(f'<text x="{gx + bar_w:.1f}" y="{top + plot_h + 18}" text-anchor="middle">{LABELS[m]}</text>')
    lx = w - 150
    for k, (metric, name) in enumerate((("acc", "Accuracy"), ("f1", "F1 (important class)"))):
        parts.append(f'<rect x="{lx}" y="{h - 40 + k * 16}" width="10" height="10" fill="{COLORS[metric]}"/>')
        parts.append(f'<text x="{lx + 14}" y="{h - 31 + k * 16}">{name}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def main(results_path: str, out_path: str) -> None:
    with open(results_path) as f:
        svg = render(json.load(f))
    with open(out_path, "w") as f:
        f.write(svg + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
