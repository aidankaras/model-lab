# Results index

Recorded outputs of each experiment. Raw transcripts and task prompts are not included (they contain private data), so these are the scored summaries.

Experiment IDs: `e` = model-comparison experiment, `t` = tooling test. Letters after the number are follow-ups to the same question.

| File | What it is |
|---|---|
| `email-triage/lora-triage.json` | Email triage study: regex rules vs TF-IDF + logistic regression vs zero-shot 1.5B vs LoRA-tuned 0.5B on 150 hand-labeled emails, with bootstrap CIs and McNemar tests. |
| `e01-e05-batch.json` | Five quick studies: cheap main-model shootout (E01), is a premium advisor model worth it (E02), compression fidelity on an 11k-word corpus (E03), free-model availability check (E04), verbosity tax derived from token counts (E05). |
| `e01b-session-benchmark.json` | 10 tasks mined from real assistant sessions, no tools. Scores honesty (fabrication) more than capability. |
| `e06-full.json` | Router test: can a cheap model route 12 user goals to the right worker? Per-model pass counts. |
| `e06-scored.json`, `e06-full-routing-accuracy.json` | Scored routing accuracy and cost per 20-scenario pass for the router test. |
| `e06-f5-spotcheck.json` | Spot check: do the auxiliary model slots still do their job after being re-pinned to a cheaper model? |
| `e07-prompt-cache.json`, `e07-parsed.json` | Does prompt caching cut cost on long sessions? Cached vs fresh input-token cost per model. |
| `e08a-reasoning-knob.json` | Does the provider's reasoning-effort parameter actually change token use and output? |
| `e08b-effort-ladder.json` | 72 runs: 3 models x 3 effort levels x 12 tasks. Cost per passed task at each level. |
| `e08b-summary.json`, `e08b-glm-summary.json` | Per-model rollups of the effort ladder. |
| `e08b-two-graders.json`, `e08b-three-graders.json` | Same answers re-graded by 2 and 3 graders from different model families. |
| `grader-robustness.json` | 5 graders scoring the same answers: how much rankings move with the grader. |
| `t01-playwright-vs-harness.json` | Raw Playwright vs an existing CDP browser harness on scripted browsing tasks. |
| `t02-bot-wall.json` | Follow-up to t01: can either tool get past a Cloudflare bot wall with default settings? (No.) |

## Model names

Names are the provider IDs used through an OpenAI-compatible router at the time of the runs: `kimi-k2.6` (Moonshot), `deepseek-v4.1-flash` / `deepseek-v4-flash` (DeepSeek), `glm-5.3-flash` (Zhipu), `gpt-6-luna` (OpenAI small tier). Prices change; the cost figures are as recorded, not current.
