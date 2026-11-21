# Results index

Recorded outputs of each experiment. Raw transcripts and task prompts are not included (they contain private data), so these are the scored summaries.

Each experiment file has the same top-level shape where it applies: `experiment`, `question`, `setup`, the recorded numbers, `finding`, `limitations` and `cost_usd_recorded` (the sum of the costs recorded in that file; some grader passes were not costed). `python -m modellab.rollup` recomputes the README's model-comparison numbers from these files.

Experiment IDs: `e` = model-comparison experiment, `t` = tooling test. Letters after the number are follow-ups to the same question.

| File | What it is |
|---|---|
| `email-triage/lora-triage.json` | Email triage study: regex rules vs TF-IDF + logistic regression vs zero-shot 1.5B vs LoRA-tuned 0.5B on 150 hand-labeled emails, with bootstrap CIs and McNemar tests. |
| `e01-e05-batch.json` | Five quick studies: cheap main-model shootout (E01), is a premium advisor model worth it (E02), compression fidelity on an 11k-word corpus (E03), free-model availability check (E04), verbosity tax derived from token counts (E05). |
| `e01b-session-benchmark.json` | 10 tasks mined from real assistant sessions, no tools. Scores honesty (fabrication) more than capability. |
| `e06a-goal-routing.json` | Router test A: can a cheap model assign 12 real goals to the right worker role? Per-model match counts, one batch each. |
| `e06b-scenario-routing.json` | Router test B (a different, harder test): route decisions (none / default / advisor) on 20 hand-labeled scenarios, 2 runs per model. Accuracy and cost per 20-scenario pass. |
| `e06c-slot-spotcheck.json` | Spot check: does a cheaper model still produce usable one-line descriptions and session titles? |
| `e07-prompt-cache.json`, `e07-parsed.json` | Does prompt caching cut cost on long sessions? Cached vs fresh input-token cost per model. |
| `e08a-reasoning-knob.json` | Does the provider's reasoning-effort parameter actually change token use and output? |
| `e08b-effort-ladder.json` | 108 runs: 6 tasks x 3 effort levels x 2 runs x 3 models (kimi and deepseek first, glm added later). Pass counts and cost per passed task, graded by one grader. |
| `e08b-summary.json`, `e08b-glm-summary.json` | Per-model rollups of the effort ladder. |
| `e08b-two-graders.json`, `e08b-three-graders.json` | Same answers re-graded by 2 and 3 graders, with a majority-of-3 column. One of the three graders (deepseek-v4-pro) shares a family with the deepseek candidate, so the deepseek cells are contaminated. |
| `grader-robustness.json` | 5 graders scoring the same answers: how much rankings move with the grader. |
| `t01-playwright-vs-harness.json` | Raw Playwright vs an existing CDP browser harness on scripted browsing tasks. |
| `t02-bot-wall.json` | Follow-up to t01: can either tool get past a Cloudflare bot wall with default settings? (No.) |

## Model names

Names are the provider IDs used through an OpenAI-compatible router at the time of the runs: `kimi-k2.6` (Moonshot), `deepseek-v4.1-flash` / `deepseek-v4-flash` (DeepSeek), `glm-5.3-flash` (Zhipu), `gpt-6-luna` (OpenAI small tier). Prices change; the cost figures are as recorded, not current.
