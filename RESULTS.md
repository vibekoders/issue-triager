# Results

Experiment runs from [PLAN.md](PLAN.md), on hosted models:

- **GPT-6 Luna** (Azure OpenAI) for the strategies' LLM calls.
- **GPT-6 Sol** for the larger model in m2, the ensemble's judge aggregator, and the eval judge for quality and suggested reply.
- **TypeSafe Jev** for the decision-model calls.
- Local tev1:0.8b on Ollama, only as the decision-model comparison in m3.

The definitions were in `evals/experiments-hosted.yaml`, now the `hosted` profile in [evals/experiments/](evals/experiments/README.md) (one file per experiment). Run 2026-10-14, with strategy versions Heuristic 3, SingleShot 3, Workflow 4, Agent 4, Ensemble 3 and DraftVerify 4.

**Read these as a demonstration of the loop, not as conclusions.** Model-based runs use 20 issues each (kubernetes-dev-20 for experiments 2–6, kubernetes-test-20 for m7). The 95% intervals on per-field accuracy are ±15–20 points, and only a few of the differences below are statistically significant (paired tests, noted where they are). Re-running the same config moved composites by up to ±1.5 points; part of that is the judge.

The experiments have since been renamed from m1–m7, which the tables below still use: m1 → `heuristic-base-rates`, m2 → `single-shot-model-size`, m3 → `workflow-escalation`, m4 → `agent-tools`, m5 → `ensemble-voters`, m6 → `escalation-coordination`, m7 → `final-comparison`.

Reproduce with (from the repo root; the hosted profile is the default):

```sh
devenv shell -- bash -c 'cd evals && uv run run-experiments --size 20'
```

The reports land in `evals/results/experiments/20-hosted/` (gitignored): one per experiment, plus `overview.md`. Run ids are now composed as `{step id}-{split}-{size}-{profile}`, so new runs don't reuse the results these numbers come from. Those (run ids `h2m*` and `m1-*`, and their reports) are archived in `evals/results/archive-before-profiles/`. m1 then ran on dev-200 and the whole test split; it now runs on the same dev/test sets as the other experiments.

**What it cost:** $1.48 in strategy calls plus $2.09 for the Sol judge, about **$3.60** for the full set. Wall time was about 45 minutes with 8 cases in parallel. The Agent and DraftVerify test runs took the longest, about 11 minutes each.

## Final comparison (m7, test-20)

Composite is a 0–100 weighted score across fields (see [evals/suite.yaml](evals/suite.yaml)). A field the strategy didn't produce counts as 0.

| Strategy | Config | Composite | Kind macro-F1 | Cost/issue | Latency p50 | Pareto |
|---|---|---|---|---|---|---|
| Heuristic v3 | no models | 62.3 | 48.6 | $0 | 0 ms | frontier |
| Workflow, Jev only | `workflow-jev.yaml` | 63.4 | 65.3 | $0.0001 | 0.2 s | **frontier** |
| Workflow, Jev + Luna escalation | `workflow-jev-escalate-0.8.yaml` | 62.1 | 70.7 | $0.0009 | 9.7 s | dominated |
| DraftVerify, Jev + Luna | `draft-verify-jev.yaml` | 63.3 | 73.2 | $0.0013 | 9.4 s | dominated |
| Agent, Luna + tools | `agent-luna.yaml` | 70.0 | 70.5 | $0.0044 | 20.7 s | **frontier** |
| SingleShot, Sol (won m2) | `single-shot-sol.yaml` | **73.0** | 73.2 | $0.0088 | 5.7 s | **frontier** |
| Ensemble, Luna + Sol + Jev, vote | `ensemble-hosted-vote.yaml` | 69.2 | 70.7 | $0.0096 | 12.5 s | dominated |

Significant differences (exact McNemar on per-case composite wins):
- **SingleShot vs heuristic:** +10.8 points, p = 0.004.
- **Agent vs heuristic:** +7.8 points, p = 0.012.
- **Ensemble vs heuristic:** +6.9 points, p = 0.012.
- **Jev-only workflow vs heuristic:** +1.1 points, p = 0.035.

Escalation vs no escalation (p = 0.69) and DraftVerify vs Workflow (p = 1.0) are not distinguishable at this size.

**Caveat on reply:** Workflow and DraftVerify were run without a reply model, and the composite counts a missing `suggested_reply` as 0. Leaving the reply out of the composite lifts both by about 6 points, to about 69. The ranking doesn't change.

## Experiments

| | Change | Composite | Cost/issue | What it shows |
|---|---|---|---|---|
| m1 | heuristic v3 → v3 + base rates (dev-200 / test-580) | 64.1 → 69.3 / 64.0 → 68.2 | $0 | The loop at $0. Base rates win because the composite counts unanswered fields as 0: guessing a field beats leaving it empty. |
| m2 | single_shot Luna → Sol | 74.2 → 77.3 | $0.0005 → $0.0092 | +3 points for 17× the cost. Luna is the better value; Sol is better outright. |
| m3 | workflow (Jev) no escalation → escalation at threshold 0.6 / 0.8 / 0.9 | 68.8 → 70.8 / 69.3 / 70.7 | $0.0001 → $0.0004 / $0.0008 / $0.0011 | Jev alone gets most of the way for a tenth of a cent. Escalation adds about 1–2 points, and a higher threshold costs more without helping. |
| m3 | Jev → local tev1:0.8b | 68.8 → 57.3 | $0.0001 → $0.0009 (shadow price; 3 split calls) | The tiny local decision model is clearly worse (−11.5). Model size matters more for decision models than escalation does. |
| m4 | agent without tools → with tools | 75.1 → 76.4 | $0.0006 → $0.0042 | Tools cost 7× and help a little (area 95% on test). The fields that tools would help most, duplicates, have no labeled cases in the 20-issue sets. |
| m5 | ensemble 1 voter → 3 voters, vote → 3 voters, judge aggregator | 72.8 → 74.0 → 79.0 | $0.0005 → $0.0096 → $0.0194 | The vote alone barely helps; letting Sol decide over the votes does (+5). At 2× the cost of SingleShot Sol, this is the best composite on dev. |
| m6 | workflow with escalation → draft_verify (both at 0.8) | 69.3 → 69.0 | $0.0008 → $0.0013 | Agent-coordinated escalation costs more and is no better than workflow-coordinated escalation here. |

## Findings

- **SingleShot is hard to beat.** One structured call to a good model is on the Pareto frontier at the top. Every orchestration pattern either costs more for less, or costs less for less.
- **A decision model alone is the cheap end of the frontier.** Jev answers the typed fields in about 0.2 s for $0.0001 per issue, roughly 90× cheaper than SingleShot on Sol, about 10 points lower on the composite, and better than the heuristic.
- **Escalation hasn't paid off yet.** Targeted Luna calls for Jev's low-confidence fields add cost and latency, but no significant quality. Jev's confidence may not separate right from wrong answers well enough to route on. The calibration sections in `m3.md` and `m7.md` are where to look next.
- **Agents help where context matters.** The agent with tools has the best area accuracy, at 95% on test. It is also the slowest, at about 20 s per issue.
- **The weak spots are shared.** Severity is 0/4 for every strategy on test-20, and regression and security have no positive cases at this size. Those fields need the larger datasets to say anything.

## Issues found along the way

These were fixed during the runs, and they show why the eval loop is worth having:

- **Silent truncation on reasoning models.** The output caps tuned for local Qwen (400 tokens for escalation) were used up by GPT-6's reasoning. 26 escalations in the first hosted run failed silently and fell back to the Jev draft, so escalation looked like it changed nothing. The caps are now ceilings with room for reasoning, and those strategy versions were bumped. The first run is archived in `evals/results/archive-hosted-run1/`.
- **GPT-6 on Azure `/chat/completions` rejects function tools unless reasoning is off.** Tool-using calls to Azure now send `reasoning_effort: none`. The judge uses the Responses API instead.
- **Answer leaks found in review:** retitled issues, `/committee security-response` lines in bodies, and corpus results that mentioned later issues. All fixed before these runs (see the Deviations section of PLAN.md).

## Next steps

- Run the bigger version, `run-experiments --size 500` (dev-500 for the development experiments, test-500 for the final comparison). That is 25 times the cases of the 20-issue run, so expect roughly 25 times its cost and time (`--dry-run` estimates the calls and time), and it gives intervals tight enough to separate the middle of the table.
- Give Workflow a reply model (`reply_model: azure:gpt-6-luna`), so the composite compares like with like.
- Check Jev's calibration (the reliability bins in the reports) before tuning escalation thresholds further.
- Hand-label `evals/gold/quality.yaml` and run `compare-runs judge-agreement`, to see how far the Sol judge can be trusted on quality and reply.
