# Model Lab

[![CI](https://github.com/aidankaras/model-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/aidankaras/model-lab/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

Small, cheap, honest evaluations of which model is worth its price for a given job, run on real workloads instead of leaderboards. Includes a negative-result-friendly write-up of what held up and what did not.

## Problem

Benchmarks measure the wrong thing for a personal assistant stack: a model can top a leaderboard and still be verbose enough to double output cost, weak at tool use, or prone to confident falsehoods. The question here is narrower: **which model gives the best value for each job, measured on tasks like the ones actually run?**

## Method

- **Cost per passed task.** Each task is pass/fail; the headline number is total spend divided by tasks passed (`modellab/scoring.py`). Zero passes means the number is undefined, not zero.
- **Cross-family graders.** The grader is never the candidate and never from the candidate's family (`grader_conflicts` / `assert_grader_separate`). Where this was violated in an early run, the write-up flags the affected cells as contaminated.
- **Fabrication = automatic fail.** A task passes only if the grader passes it and there are zero fabricated claims.
- **Grader robustness.** After finding that rankings moved with the grader, single-grader "A beats B" claims are treated as indicative only; majority of three graders from different families is the bar.
- Every experiment states an expected cost before running; the total spend across everything in `results/` is a few dollars at most.

## Headline results

| Experiment | Finding | Source |
|---|---|---|
| Reasoning effort ladder (72 runs, 3 models) | No effort level buys a clear quality win. Cost per passed task: kimi-k2.6 $0.0129, deepseek-v4.1-flash $0.0015, glm-5.3-flash $0.0009 (0 fabrications). | `results/e08b-effort-ladder.json` |
| Grader robustness (5 graders, same answers) | Grader spread was 6-9 of 12 points on most cells; a model's apparent lead depended on which grader was used. glm had the lowest spread (2-4). | `results/grader-robustness.json` |
| Router test (12 goals, 3 models) | gpt-6-luna 12/12, glm-5.3-flash 12/12, deepseek-v4-flash 11/12. One rater, one batch each. | `results/e06-full.json` |
| Session-mined benchmark (10 tasks, no tools) | glm 10/10, deepseek 10/10, kimi 7/10 (three fabricated-adjacent answers). Tests honesty, not capability. | `results/e01b-session-benchmark.json` |
| Prompt caching, long sessions | Caching works on glm-5.3-flash but saves only 21-36% of input cost because input is already cheap; cached vs fresh input price ratio is 1.56x on glm vs 6.7x on deepseek. | `results/e07-prompt-cache.json` |
| Bot-wall browsing test | Negative: neither tool got past a Cloudflare-protected page with defaults. | `results/t02-bot-wall.json` |

### Email triage (small local model vs. rules)

Binary "needs attention / skip" on 150 held-out real emails hand-labeled by the author (71 important / 79 not). Data not released for privacy. Cost $0 (CPU only).

| System | Acc | Precision | Recall | F1 | p50 latency | p95 latency |
|---|---|---|---|---|---|---|
| Production regex rules | .533 | 1.000 | .014 | .028 | n/a | n/a |
| LoRA-tuned Qwen2.5-0.5B (263 training emails) | .700 | .691 | .662 | .676 | 111 ms | 168 ms |
| Zero-shot Qwen2.5-1.5B-Instruct | .460 | .464 | .915 | .616 | 1443 ms | 1846 ms |

The rules catch one of 71 important emails. The zero-shot model gets high recall only by calling nearly everything important (75 false positives, 4 true negatives), so its accuracy is below the 52.7% majority-class rate. The LoRA model is the only one that is a usable trade-off. Source: `results/email-triage/results-lora-triage.json`.

## How to reproduce

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest -q          # offline, no network, no model downloads
ruff check .
```

Demo on the synthetic fixture (20 fake emails, all `example.com` / `acme.com`):

```bash
python -m email_triage.make_results tests/fixtures/emails_synth.jsonl /tmp/none /tmp/out.json
```

To retrain the email classifier on your own mailbox: `pip install -r requirements-train.txt`, produce a JSONL in the fixture schema (`tid,name,from,subject,snippet,date,labels,split,gold,prod`) and a `tid<TAB>0|1` label file for the train rows, then run `python -m email_triage.triage_binary lora DATA.jsonl LABELS.tsv OUT_DIR`. `email_triage/pull.py` shows one way to fetch metadata through the Gmail API.

The per-experiment JSONs in `results/` are the recorded outputs. The raw transcripts and task prompts are not included, so the model-comparison experiments are documented, not re-runnable from this repo.

## Limits

- Single labeler for the email gold set; n = 150; the 71/79 class split gives wide confidence intervals.
- 263 of the training labels were produced by an LLM from a written rubric, not by a human.
- CPU-only training and inference; latencies are for one specific machine.
- Model-comparison experiments use n = 2 runs per cell, small task sets (6-12), and in several cases one rater. They are indicative, not publishable.
- The email data (and the raw transcripts behind the other experiments) are not released.
- Model names and prices in the result files are as of October 2026 and will go stale.

## License

MIT, see [LICENSE](LICENSE).
