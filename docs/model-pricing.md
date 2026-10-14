# Model pricing estimates

> [!WARNING]
> **These are estimates, not measurements.** Only GPT-6 Luna and Sol have
> been run on this task. Every other column reuses their token counts at
> another model's list price, and real costs can be considerably higher (see
> [Caveats](#caveats)). For measured costs and run times per experiment, see
> [costs.md](costs.md).

What a run costs for each strategy and dataset size with GPT-6 Luna, GPT-5.6
Terra, GPT-6 Sol, Claude Sonnet 5.5, Claude Opus 5.5 and Gemini 4 Argon.
Prices are list prices as of 2026-10-14. Cells give **Global / Data Zone**
prices; Gemini isn't on Azure, so it has one price.

## Prices

USD per million tokens, input / output, short context:

| Model | Global | Data Zone |
|---|---|---|
| GPT-6 Luna (Azure) | $0.10 / $0.50 | $0.12 / $0.60 |
| GPT-5.6 Terra (Azure) | $2.00 / $12.00 | $2.20 / $13.20 |
| GPT-6 Sol (Azure) | $2.00 / $10.00 | $2.40 / $12.00 |
| Claude Sonnet 5.5 (Microsoft Foundry) | $2.00 / $10.00 | $2.20 / $11.00 |
| Claude Opus 5.5 (Microsoft Foundry) | $4.00 / $20.00 | $4.40 / $22.00 |
| Gemini 4 Argon (Gemini API) | $4.00 / $20.00 | – |

- **GPT-6:** there is no GPT-6 Terra; Terra is a GPT-5.6 model. Azure's Global Flex tier
  (GPT-5.6 Terra at $1 / $6) is left out.
- **Claude on Foundry** is billed at Anthropic's standard rates. Its US Data
  Zone Standard deployment applies a 1.1× multiplier.
- **Gemini 4 Argon** is at its introductory price of $2 / $10 until
  2026-11-30. The table uses the $4 / $20 price that applies from 2026-12-01.
- **Jev** (`typesafe:jev-latest`), the decision model in Workflow and
  DraftVerify, is $0.042 / $0 (output is free), about $0.0001 per issue. It's
  included in those rows.

## Tokens per issue

These are the means over the hosted 20-issue runs in
`evals/results/archive-*/` (dev-20 and test-20, two runs each). They count
only the tokens sent to the model being priced. In mixed runs, Jev's share is
the usage of the Jev-only workflow on the same issues.

| Strategy | Input | Output |
|---|---|---|
| SingleShot | 2,320 | 500 |
| Workflow, Jev + escalation at 0.6 | 1,400 | 300 |
| Workflow, Jev + escalation at 0.8 | 3,340 | 660 |
| Workflow, Jev + escalation at 0.9 | 4,730 | 890 |
| DraftVerify, Jev + this model as verifier | 8,610 | 660 |
| Agent, with tools | 36,710 | 1,130 |
| Judge, if this model judges (quality + reply) | 1,960 | 314 |

## Cost per issue

| Strategy | GPT-6 Luna | GPT-5.6 Terra | GPT-6 Sol | Sonnet 5.5 | Opus 5.5 | Gemini 4 Argon |
|---|---|---|---|---|---|---|
| SingleShot | $0.0005 / $0.0006 | $0.0106 / $0.0117 | $0.0096 / $0.0116 | $0.0096 / $0.0106 | $0.0193 / $0.0212 | $0.0193 |
| Workflow, Jev + escalation at 0.6 | $0.0004 / $0.0004 | $0.0065 / $0.0071 | $0.0059 / $0.0071 | $0.0059 / $0.0065 | $0.0117 / $0.0129 | $0.0117 |
| Workflow, Jev + escalation at 0.8 | $0.0008 / $0.0009 | $0.0147 / $0.0162 | $0.0134 / $0.0160 | $0.0134 / $0.0147 | $0.0267 / $0.0293 | $0.0267 |
| Workflow, Jev + escalation at 0.9 | $0.0010 / $0.0012 | $0.0202 / $0.0223 | $0.0185 / $0.0221 | $0.0185 / $0.0203 | $0.0368 / $0.0405 | $0.0368 |
| DraftVerify, Jev + this model as verifier | $0.0013 / $0.0015 | $0.0252 / $0.0278 | $0.0239 / $0.0287 | $0.0239 / $0.0263 | $0.0477 / $0.0525 | $0.0477 |
| Agent, with tools | $0.0042 / $0.0051 | $0.0870 / $0.0957 | $0.0847 / $0.1017 | $0.0847 / $0.0932 | $0.1694 / $0.1864 | $0.1694 |
| Judge, extra per run if this model judges | $0.0004 / $0.0004 | $0.0077 / $0.0085 | $0.0071 / $0.0085 | $0.0071 / $0.0078 | $0.0141 / $0.0155 | $0.0141 |

## Cost per dataset size

### 500 issues (dev-500, test-500)

| Strategy | GPT-6 Luna | GPT-5.6 Terra | GPT-6 Sol | Sonnet 5.5 | Opus 5.5 | Gemini 4 Argon |
|---|---|---|---|---|---|---|
| SingleShot | $0.24 / $0.29 | $5.32 / $5.85 | $4.82 / $5.78 | $4.82 / $5.30 | $9.64 / $10.60 | $9.64 |
| Workflow, Jev + escalation at 0.6 | $0.19 / $0.22 | $3.25 / $3.57 | $2.95 / $3.53 | $2.95 / $3.24 | $5.85 / $6.43 | $5.85 |
| Workflow, Jev + escalation at 0.8 | $0.38 / $0.45 | $7.35 / $8.08 | $6.69 / $8.02 | $6.69 / $7.35 | $13.33 / $14.66 | $13.33 |
| Workflow, Jev + escalation at 0.9 | $0.51 / $0.60 | $10.12 / $11.13 | $9.23 / $11.07 | $9.23 / $10.15 | $18.41 / $20.25 | $18.41 |
| DraftVerify, Jev + this model as verifier | $0.64 / $0.76 | $12.62 / $13.88 | $11.96 / $14.34 | $11.96 / $13.15 | $23.87 / $26.25 | $23.87 |
| Agent, with tools | $2.12 / $2.54 | $43.49 / $47.84 | $42.36 / $50.83 | $42.36 / $46.60 | $84.72 / $93.19 | $84.72 |
| Judge, extra per run if this model judges | $0.18 / $0.21 | $3.84 / $4.23 | $3.53 / $4.24 | $3.53 / $3.88 | $7.06 / $7.77 | $7.06 |

### 20 issues (dev-20, test-20)

| Strategy | GPT-6 Luna | GPT-5.6 Terra | GPT-6 Sol | Sonnet 5.5 | Opus 5.5 | Gemini 4 Argon |
|---|---|---|---|---|---|---|
| SingleShot | $0.01 / $0.01 | $0.21 / $0.23 | $0.19 / $0.23 | $0.19 / $0.21 | $0.39 / $0.42 | $0.39 |
| Workflow, Jev + escalation at 0.6 | $0.01 / $0.01 | $0.13 / $0.14 | $0.12 / $0.14 | $0.12 / $0.13 | $0.23 / $0.26 | $0.23 |
| Workflow, Jev + escalation at 0.8 | $0.02 / $0.02 | $0.29 / $0.32 | $0.27 / $0.32 | $0.27 / $0.29 | $0.53 / $0.59 | $0.53 |
| Workflow, Jev + escalation at 0.9 | $0.02 / $0.02 | $0.40 / $0.45 | $0.37 / $0.44 | $0.37 / $0.41 | $0.74 / $0.81 | $0.74 |
| DraftVerify, Jev + this model as verifier | $0.03 / $0.03 | $0.50 / $0.56 | $0.48 / $0.57 | $0.48 / $0.53 | $0.95 / $1.05 | $0.95 |
| Agent, with tools | $0.08 / $0.10 | $1.74 / $1.91 | $1.69 / $2.03 | $1.69 / $1.86 | $3.39 / $3.73 | $3.39 |
| Judge, extra per run if this model judges | $0.01 / $0.01 | $0.15 / $0.17 | $0.14 / $0.17 | $0.14 / $0.16 | $0.28 / $0.31 | $0.28 |

### 200 issues (dev-200)

| Strategy | GPT-6 Luna | GPT-5.6 Terra | GPT-6 Sol | Sonnet 5.5 | Opus 5.5 | Gemini 4 Argon |
|---|---|---|---|---|---|---|
| SingleShot | $0.10 / $0.12 | $2.13 / $2.34 | $1.93 / $2.31 | $1.93 / $2.12 | $3.86 / $4.24 | $3.86 |
| Workflow, Jev + escalation at 0.6 | $0.08 / $0.09 | $1.30 / $1.43 | $1.18 / $1.41 | $1.18 / $1.30 | $2.34 / $2.57 | $2.34 |
| Workflow, Jev + escalation at 0.8 | $0.15 / $0.18 | $2.94 / $3.23 | $2.68 / $3.21 | $2.68 / $2.94 | $5.33 / $5.86 | $5.33 |
| Workflow, Jev + escalation at 0.9 | $0.20 / $0.24 | $4.05 / $4.45 | $3.69 / $4.43 | $3.69 / $4.06 | $7.36 / $8.10 | $7.36 |
| DraftVerify, Jev + this model as verifier | $0.26 / $0.31 | $5.05 / $5.55 | $4.78 / $5.74 | $4.78 / $5.26 | $9.55 / $10.50 | $9.55 |
| Agent, with tools | $0.85 / $1.02 | $17.40 / $19.14 | $16.94 / $20.33 | $16.94 / $18.64 | $33.89 / $37.28 | $33.89 |
| Judge, extra per run if this model judges | $0.07 / $0.08 | $1.54 / $1.69 | $1.41 / $1.69 | $1.41 / $1.55 | $2.82 / $3.11 | $2.82 |

### 580 issues (test-600)

| Strategy | GPT-6 Luna | GPT-5.6 Terra | GPT-6 Sol | Sonnet 5.5 | Opus 5.5 | Gemini 4 Argon |
|---|---|---|---|---|---|---|
| SingleShot | $0.28 / $0.34 | $6.17 / $6.79 | $5.59 / $6.71 | $5.59 / $6.15 | $11.18 / $12.30 | $11.18 |
| Workflow, Jev + escalation at 0.6 | $0.23 / $0.26 | $3.77 / $4.14 | $3.42 / $4.09 | $3.42 / $3.76 | $6.79 / $7.46 | $6.79 |
| Workflow, Jev + escalation at 0.8 | $0.44 / $0.52 | $8.53 / $9.37 | $7.76 / $9.30 | $7.76 / $8.53 | $15.46 / $17.00 | $15.46 |
| Workflow, Jev + escalation at 0.9 | $0.59 / $0.70 | $11.74 / $12.91 | $10.71 / $12.84 | $10.71 / $11.77 | $21.35 / $23.48 | $21.35 |
| DraftVerify, Jev + this model as verifier | $0.75 / $0.89 | $14.64 / $16.10 | $13.87 / $16.64 | $13.87 / $15.25 | $27.69 / $30.45 | $27.69 |
| Agent, with tools | $2.46 / $2.95 | $50.45 / $55.49 | $49.14 / $58.97 | $49.14 / $54.05 | $98.28 / $108.10 | $98.28 |
| Judge, extra per run if this model judges | $0.20 / $0.25 | $4.46 / $4.90 | $4.09 / $4.91 | $4.09 / $4.50 | $8.19 / $9.01 | $8.19 |

## Caveats

- **Thinking models will cost more than shown.** Claude Opus 5.5 always
  thinks, at medium effort by default, and Gemini bills thinking tokens as
  output. Their output could be several times the 300–1,130 tokens measured
  on GPT-6, and output is most of the cost. Treat the Opus 5.5 and Gemini
  columns as a lower bound.
- **Tokenizers differ.** Claude's tokenizer can produce up to about 1.35× as
  many tokens for the same text, which would raise the Claude columns by up
  to about a third.
- **The Agent rows assume no prompt caching.** The agent resends its context
  on each of about 7 turns. With caching, its input cost could fall a lot
  (cached input is $0.20 per million on Sol and Terra, $0.10 on Sonnet 5.5).
- **DraftVerify is approximate.** Jev's share of its input is assumed to equal
  the Jev-only workflow's, about 2,350 tokens per issue.
- **Ensemble is left out.** It mixes models (Luna, Sol and Jev voters), so it
  doesn't fit a single column. Measured on GPT-6 it costs about $0.0096 per
  issue (see [costs.md](costs.md)).
- **One token profile is used for every model.** Measured Luna and Sol runs
  differ a little from these rows: Sol wrote fewer output tokens and Luna
  more.
- **20 issues is a small sample.** Token means come from 20-issue runs, and a
  repeat of the same config moved them by up to about 15%.
- **Prices change.** Check them against the price pages before relying on
  them.

## Sources

- [Azure OpenAI pricing](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/)
- [Claude pricing](https://platform.claude.com/docs/en/about-claude/pricing),
  including Claude in Microsoft Foundry and data residency pricing
- [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing)
