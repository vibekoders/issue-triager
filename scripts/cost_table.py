"""Cost and time tables for docs/costs.md, from measured results files.

Reads the hosted runs (GPT-6 Luna/Sol, TypeSafe Jev, the GPT-6 Sol judge) in
evals/results/archive-before-profiles/h2m*/, takes per-case means of the
strategy's usage (tokens, calls, latency), the judge's usage and the wall
time, and prices them for other models and provider profiles. Prints
Markdown. Standard library only:

    python3 scripts/cost_table.py            # from the repo root
    python3 scripts/cost_table.py --results evals/results/archive-before-profiles

How the numbers are derived (also stated in docs/costs.md):

- Usage in a results file is the sum over every model a case called, so runs
  that mix models are split by role: the Jev draft of a workflow is the
  Jev-only workflow's usage on the same cases, the escalation/verifier part is
  the rest; an ensemble's llm voter is the 1-voter run, its Jev voter the
  Jev-only workflow, and llm_large (Sol voter, plus the judge aggregator) the
  rest. The check column compares the split, re-priced at GPT-6 prices, with
  the measured cost_usd.
- Token counts measured on GPT-6 (Luna/Sol, with their reasoning tokens) are
  reused for every other hosted model. Other tokenizers and reasoning budgets
  move them by roughly +-30%.
- Hosted time: the measured per-case duration (strategy) plus the judge's
  time per case (derived from each run's wall time at 8 cases in parallel),
  divided by 8. Steps that use an Ollama model run one case at a time.
- Local (Ollama, CPU only): calls x seconds per call; tev1:0.8b's 3-call draft
  is measured (h2m3-workflow-tev1-none), the rest are the per-call figures in
  evals/experiments/profiles.yaml (qwen3:1.7b 30 s, matching the measured
  median of verify-singleshot-1; qwen3:4b 60 s; tev1:4b 15 s).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from dataclasses import dataclass, field
from datetime import datetime

# USD per million tokens (input, output). Keep in step with
# crates/triage-core/src/price.rs and the providers' price pages.
PRICES = {
    "gpt-6-luna": (0.10, 0.50),
    "gpt-6-sol": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-5": (2.0, 10.0),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.1-pro-preview": (2.0, 12.0),
    "jev": (0.042, 0.0),  # typesafe:jev-latest, output free
    # Local Ollama models cost nothing to call. (price.rs gives them shadow
    # prices so the reports' cost axis means something; not used here.)
    "qwen3:1.7b": (0.0, 0.0),
    "qwen3:4b": (0.0, 0.0),
    "tev1:0.8b": (0.0, 0.0),
    "tev1:4b": (0.0, 0.0),
}

HOSTED_MODELS = [
    "gpt-6-luna", "gpt-6-sol", "claude-haiku-4-5", "claude-sonnet-5",
    "gemini-3.5-flash-lite", "gemini-3.1-pro-preview",
]
# The larger model of each hosted model's provider (the ensemble's llm_large
# when the llm column is that model).
LARGE_OF = {
    "gpt-6-luna": "gpt-6-sol", "gpt-6-sol": "gpt-6-sol",
    "claude-haiku-4-5": "claude-sonnet-5", "claude-sonnet-5": "claude-sonnet-5",
    "gemini-3.5-flash-lite": "gemini-3.1-pro-preview",
    "gemini-3.1-pro-preview": "gemini-3.1-pro-preview",
}

PROFILES = {
    "openai": dict(llm="gpt-6-luna", llm_large="gpt-6-sol", decision="jev",
                   decision_alt="tev1:0.8b", judge="gpt-6-sol", judge_cheap="gpt-6-luna"),
    "anthropic": dict(llm="claude-haiku-4-5", llm_large="claude-sonnet-5", decision="jev",
                      decision_alt="tev1:0.8b", judge="claude-sonnet-5",
                      judge_cheap="claude-haiku-4-5"),
    "gemini": dict(llm="gemini-3.5-flash-lite", llm_large="gemini-3.1-pro-preview",
                   decision="jev", decision_alt="tev1:0.8b", judge="gemini-3.1-pro-preview",
                   judge_cheap="gemini-3.5-flash-lite"),
    "local": dict(llm="qwen3:1.7b", llm_large="qwen3:4b", decision="tev1:0.8b",
                  decision_alt="tev1:4b", judge="qwen3:4b", judge_cheap="qwen3:4b"),
}

# Local CPU seconds per call (4 CPUs, 8 GB RAM, no GPU). "tev1:0.8b" is per
# 3-call draft, measured; the others are profiles.yaml's per-call figures.
LOCAL_SECONDS = {"qwen3:1.7b": 30.0, "qwen3:4b": 60.0, "tev1:0.8b": None, "tev1:4b": 45.0}
CONCURRENCY = 8
SIZES = [20, 200, 500, 580]


@dataclass
class Run:
    name: str
    n: int
    tin: float
    tout: float
    llm: float
    dec: float
    tool: float
    latency: float  # s, the service's summed model/tool time
    duration: float  # s, the harness's per-case time
    cost: float
    jin: float
    jout: float
    jreq: float
    wall: float  # s, whole run

    @property
    def judge_seconds(self) -> float:
        """Judge time per case: per-case wall at 8 in parallel minus the strategy."""
        conc = 1 if self.name == "h2m3-workflow-tev1-none-dev-20" else CONCURRENCY
        return self.wall * conc / self.n - self.duration


def load(results: str) -> dict[str, Run]:
    runs = {}
    for path in sorted(glob.glob(os.path.join(results, "*", "*.json"))):
        name = os.path.basename(os.path.dirname(path))
        d = json.load(open(path))
        cases = [c for c in d["cases"] if c.get("output")]
        if not cases:
            continue
        u = [c["output"]["usage"] for c in cases]
        j = [c.get("judge_usage") or {} for c in cases]
        mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
        m = d["metadata"]
        wall = (datetime.fromisoformat(m["finished_at"]) - datetime.fromisoformat(m["started_at"])).total_seconds()
        runs[name] = Run(
            name, len(cases),
            mean([x["input_tokens"] for x in u]), mean([x["output_tokens"] for x in u]),
            mean([x["llm_calls"] for x in u]), mean([x["decision_calls"] for x in u]),
            mean([x["tool_calls"] for x in u]), mean([x["latency_ms"] for x in u]) / 1000,
            mean([c["duration"] for c in cases]), mean([x["cost_usd"] for x in u]),
            mean([x.get("input_tokens", 0) for x in j]), mean([x.get("output_tokens", 0) for x in j]),
            mean([x.get("requests", 0) for x in j]), wall,
        )
    return runs


@dataclass
class Part:
    role: str  # llm, llm_large, decision, decision_alt
    tin: float
    tout: float
    calls: float


@dataclass
class Step:
    id: str
    split: str
    label: str
    run: Run
    parts: list[Part] = field(default_factory=list)

    def cost(self, models: dict[str, str]) -> float:
        total = 0.0
        for p in self.parts:
            pin, pout = PRICES[models[p.role]]
            total += (p.tin * pin + p.tout * pout) / 1e6
        return total

    def judge_cost(self, judge: str) -> float:
        pin, pout = PRICES[judge]
        return (self.run.jin * pin + self.run.jout * pout) / 1e6

    @property
    def ollama_in_hosted(self) -> bool:
        return any(p.role == "decision_alt" for p in self.parts)

    def local_seconds(self, models: dict[str, str], tev1_draft: float) -> float:
        s = 0.0
        for p in self.parts:
            m = models[p.role]
            if m == "tev1:0.8b":
                s += tev1_draft
            elif m == "tev1:4b":
                s += LOCAL_SECONDS[m]
            else:
                s += p.calls * LOCAL_SECONDS[m]
        return s


def build_steps(r: dict[str, Run]) -> list[Step]:
    """Every step of evals/experiments/*.yaml, mapped to the measured run of its config."""

    def single(role, run):
        return [Part(role, run.tin, run.tout, max(run.llm, 1.0))]

    def jev(split):
        return r[f"h2m3-workflow-none-{split}-20" if split == "dev" else "h2m7-workflow-none-test-20"]

    def with_jev(run, split):
        j = jev(split)
        return [Part("decision", j.tin, j.tout, 1), Part("llm", run.tin - j.tin, run.tout - j.tout, run.llm)]

    voter = r["h2m5-ensemble-1-voter-dev-20"]

    def ensemble(run, split, judge_agg):
        j = jev(split)
        return [
            Part("llm", voter.tin, voter.tout, 1),
            Part("decision", j.tin, j.tout, 1),
            Part("llm_large", run.tin - voter.tin - j.tin, run.tout - voter.tout - j.tout, 2 if judge_agg else 1),
        ]

    h = r["h2m7-heuristic-test-20"]
    hb = r["h2m7-heuristic-base-rates-test-20"]
    S = []
    add = lambda *a: S.append(Step(*a))  # noqa: E731
    # heuristic-base-rates: two steps x two splits (measured on test-20; dev is alike)
    add("heuristic", "dev", "Heuristic v3", h, [])
    add("heuristic", "test", "Heuristic v3", h, [])
    add("heuristic-base-rates", "dev", "Heuristic v3 + base rates", hb, [])
    add("heuristic-base-rates", "test", "Heuristic v3 + base rates", hb, [])
    # single-shot-model-size
    x = r["h2m2-single-shot-luna-dev-20"]
    add("single-shot-llm", "dev", "SingleShot, llm", x, single("llm", x))
    x = r["h2m2-single-shot-sol-dev-20"]
    add("single-shot-llm-large", "dev", "SingleShot, llm_large", x, single("llm_large", x))
    # workflow-escalation
    x = r["h2m3-workflow-none-dev-20"]
    add("workflow-decision-only", "dev", "Workflow, Jev only", x, [Part("decision", x.tin, x.tout, 1)])
    for t in ("0.8", "0.6", "0.9"):
        x = r[f"h2m3-workflow-esc-{t}-dev-20"]
        add(f"workflow-esc-{t}", "dev", f"Workflow, Jev + llm escalation at {t}", x, with_jev(x, "dev"))
    x = r["h2m3-workflow-tev1-none-dev-20"]
    add("workflow-decision-alt", "dev", "Workflow, decision_alt only (tev1 on Ollama)", x,
        [Part("decision_alt", x.tin, x.tout, 3)])
    # agent-tools
    x = r["h2m4-agent-notools-dev-20"]
    add("agent-no-tools", "dev", "Agent, no tools", x, single("llm", x))
    x = r["h2m4-agent-tools-dev-20"]
    add("agent-tools", "dev", "Agent, with tools", x, single("llm", x))
    # ensemble-voters
    add("ensemble-1-voter", "dev", "Ensemble, 1 voter", voter, single("llm", voter))
    x = r["h2m5-ensemble-vote-dev-20"]
    add("ensemble-vote", "dev", "Ensemble, llm + llm_large + Jev, vote", x, ensemble(x, "dev", False))
    x = r["h2m5-ensemble-judge-dev-20"]
    add("ensemble-judge", "dev", "Ensemble, 3 voters, llm_large judge", x, ensemble(x, "dev", True))
    # escalation-coordination
    x = r["h2m6-draft-verify-dev-20"]
    add("draft-verify", "dev", "DraftVerify, Jev + llm verifier", x, with_jev(x, "dev"))
    # final-comparison (test)
    add("final-heuristic", "test", "Heuristic v3 (derived, judge only)", h, [])
    add("final-heuristic-base-rates", "test", "Heuristic + base rates (derived, judge only)", hb, [])
    x = r["h2m7-single-shot-test-20"]
    add("final-single-shot", "test", "SingleShot, best of the two (llm_large won)", x, single("llm_large", x))
    x = r["h2m7-workflow-none-test-20"]
    add("final-workflow-decision-only", "test", "Workflow, Jev only", x, [Part("decision", x.tin, x.tout, 1)])
    x = r["h2m7-workflow-esc-0.8-test-20"]
    add("final-workflow-esc-0.8", "test", "Workflow, Jev + llm escalation at 0.8", x, with_jev(x, "test"))
    x = r["h2m7-agent-tools-test-20"]
    add("final-agent-tools", "test", "Agent, with tools", x, single("llm", x))
    x = r["h2m7-ensemble-vote-test-20"]
    add("final-ensemble-vote", "test", "Ensemble, vote", x, ensemble(x, "test", False))
    x = r["h2m7-draft-verify-test-20"]
    add("final-draft-verify", "test", "DraftVerify", x, with_jev(x, "test"))
    return S


def money(x: float) -> str:
    if x == 0:
        return "$0"
    if x < 0.01:
        return f"${x:.4f}"
    if x < 10:
        return f"${x:.2f}"
    return f"${x:,.0f}"


def per1k(x: float) -> str:
    """Cost of 1,000 cases."""
    return money(x * 1000)


def dur(s: float) -> str:
    if s < 90:
        return f"{s:.0f} s"
    if s < 90 * 60:
        return f"{s / 60:.0f} min"
    if s < 48 * 3600:
        return f"{s / 3600:.1f} h"
    return f"{s / 86400:.1f} days"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default="evals/results/archive-before-profiles")
    ap.add_argument("--local-run", default="evals/results/archive-pre-leak-fixes")
    args = ap.parse_args()
    runs = load(args.results)
    steps = build_steps(runs)
    tev1_draft = runs["h2m3-workflow-tev1-none-dev-20"].duration
    hosted_judge_s = statistics.median(
        x.judge_seconds for x in runs.values() if x.name.startswith("h2m") and x.jreq > 0
    )
    gpt = PROFILES["openai"]

    print("<!-- generated by scripts/cost_table.py; edit the script, not this block -->\n")
    print(f"Inputs: {len(runs)} results files in `{args.results}`; "
          f"hosted judge time per case (median, derived from wall time): {hosted_judge_s:.1f} s; "
          f"tev1:0.8b draft on CPU (3 calls, measured): {tev1_draft:.1f} s.\n")

    # (b) per size x profile
    def plan(profile: str, judge: str | None):
        """(cost per case summed over all steps, seconds per case summed over all steps)."""
        models = PROFILES[profile]
        c = t = 0.0
        for s in steps:
            c += s.cost(models) + (s.judge_cost(judge) if judge else 0.0)
            if profile == "local":
                sec = s.local_seconds(models, tev1_draft)
                if judge:
                    sec += s.run.jreq * LOCAL_SECONDS["qwen3:4b"]
                t += sec
            else:
                sec = s.run.duration + (hosted_judge_s if judge else 0.0)
                t += sec if s.ollama_in_hosted else sec / CONCURRENCY
        return c, t

    by_id = {(s.id, s.split): s for s in steps}
    ss = by_id[("single-shot-llm", "dev")]
    wf = by_id[("workflow-esc-0.8", "dev")]
    print("### Per dataset size and provider profile\n")
    print(f"All experiments = the {len(steps)} step runs of `evals/experiments/*.yaml` at that size "
          "(each on N cases: dev-N for the development experiments, test-N for final-comparison). "
          "`run-experiments` takes `--size 20` or `--size 500`; the 200 and 580 rows are the same "
          "plan at that many cases (cost and time are linear in cases). Single runs are "
          "strategy-only (no judge).\n")
    hdr = ["Profile", "Cases", "SingleShot (llm)", "Workflow, decision + llm escalation 0.8",
           "All experiments, no judge", "All experiments + cheap judge", "All experiments + large judge"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for profile, models in PROFILES.items():
        local = profile == "local"
        c0, t0 = plan(profile, None)
        cc, tc = plan(profile, models["judge_cheap"])
        cl, tl = plan(profile, models["judge"])
        if local:
            ss_c, ss_t = 0.0, ss.local_seconds(models, tev1_draft)
            wf_c, wf_t = 0.0, wf.local_seconds(models, tev1_draft)
        else:
            ss_c, ss_t = ss.cost(models), ss.run.duration / CONCURRENCY
            wf_c, wf_t = wf.cost(models), wf.run.duration / CONCURRENCY
        for n in SIZES:
            row = [f"{profile} ({models['llm']} / {models['llm_large']})" if n == SIZES[0] else "", str(n),
                   f"{money(ss_c * n)} · {dur(ss_t * n)}", f"{money(wf_c * n)} · {dur(wf_t * n)}",
                   f"{money(c0 * n)} · {dur(t0 * n)}"]
            if local:
                row += ["–", f"{money(cl * n)} · {dur(tl * n)}"]
            else:
                row += [f"{money(cc * n)} · {dur(tc * n)}", f"{money(cl * n)} · {dur(tl * n)}"]
            print("| " + " | ".join(row) + " |")
    print("\nCheap judge = the profile's `llm` model (Luna, Haiku, Flash-Lite), large judge = its "
          "`llm_large` (Sol, Sonnet, Pro). Local judge = qwen3:4b. Hosted profiles keep Jev as the "
          "decision model and the local tev1:0.8b as `decision_alt` (one case at a time, "
          f"{tev1_draft:.0f} s per case: about {dur(tev1_draft * 500)} of every 500-case run).\n")

    # measured totals of the 20-issue hosted run, for comparison
    h2 = [x for x in runs.values() if x.name.startswith("h2m")]
    strat = sum(x.cost * x.n for x in h2)
    print(f"Measured, hosted 20-issue run (the {len(h2)} h2m* results files): strategy "
          f"{money(strat)}, judge {money(sum(x.n * (x.jin * 2 + x.jout * 10) / 1e6 for x in h2))} "
          f"(Sol), summed step wall time {dur(sum(x.wall for x in h2))}.\n")

    # (a) per strategy, per case
    distinct = []
    seen = set()
    for s in steps:
        if s.id.startswith("final-") or s.id in seen or s.id == "heuristic-base-rates":
            continue
        seen.add(s.id)
        distinct.append(s)
    print("### Per strategy, per case (strategy only, no judge)\n")
    print("Tokens and calls are means over 20 cases (dev-20), measured on GPT-6 Luna (Sol for "
          "`llm_large`) and Jev. Cost per **1,000 cases**, with the `llm` role on the column's "
          "model and Jev as the decision model; for the ensemble rows `llm_large` is the same "
          "provider's large model (Sol, Sonnet, Pro).\n")
    hdr = ["Step", "Tokens in / out", "Calls llm / decision / tool", "Latency", "Check"] + HOSTED_MODELS + ["Ollama, CPU time"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for s in distinct:
        r = s.run
        check = s.cost(gpt)
        ok = "–" if r.cost == 0 or check == 0 else f"{check / r.cost * 100:.0f}%"
        cells = []
        for m in HOSTED_MODELS:
            models = dict(llm=m, llm_large=LARGE_OF[m] if s.id != "single-shot-llm-large" else m,
                          decision="jev", decision_alt="tev1:0.8b")
            cells.append(per1k(s.cost(models)))
        cells.append(dur(s.local_seconds(PROFILES["local"], tev1_draft)) if s.parts else "0 s")
        print(f"| `{s.id}` {s.label} | {r.tin:,.0f} / {r.tout:,.0f} | {r.llm:.1f} / {r.dec:.1f} / {r.tool:.1f} "
              f"| {r.latency:.1f} s | {ok} | " + " | ".join(cells) + " |")
    print("\nCheck: the role split re-priced at GPT-6/Jev prices, as a share of the measured `cost_usd` "
          "(100% = the split adds up). `single-shot-llm-large` is priced with the column's model as "
          "the only model. `workflow-decision-alt` runs tev1 on Ollama: $0 here. Ollama column: $0, "
          "time per case on the CPU machine with qwen3:1.7b as `llm`, qwen3:4b as `llm_large` and "
          "tev1:0.8b as the decision model (calls x seconds per call; estimate except tev1).\n")

    # (c) judge
    print("### The judge\n")
    print("Measured judge usage (GPT-6 Sol as judge): a strategy that writes a `suggested_reply` gets "
          "two judge requests (quality + reply), one without (Heuristic, Workflow, DraftVerify). "
          "Cost per **1,000 cases**. Judge tokens were measured with Sol as judge; Luna as judge "
          "(the heuristic runs judged by Luna) used about the same input and 30–50% more output.\n")
    with_reply = [x for x in runs.values() if x.name.startswith("h2m") and round(x.jreq) == 2]
    without = [x for x in runs.values() if x.name.startswith("h2m") and round(x.jreq) == 1]
    jr = [(f"with reply (2 requests, {len(with_reply)} runs)", with_reply),
          (f"without reply (1 request, {len(without)} runs)", without)]
    print("| Judge requests | Tokens in / out | " + " | ".join(HOSTED_MODELS) + " |")
    print("|---|---|" + "---|" * len(HOSTED_MODELS))
    for label, rs in jr:
        tin = statistics.mean(x.jin for x in rs)
        tout = statistics.mean(x.jout for x in rs)
        cells = [per1k((tin * PRICES[m][0] + tout * PRICES[m][1]) / 1e6) for m in HOSTED_MODELS]
        print(f"| {label} | {tin:,.0f} / {tout:,.0f} | " + " | ".join(cells) + " |")
    print("\nOn GPT-6 a Sol judge costs more per case than most strategies (SingleShot on Luna is "
          "$0.54 per 1,000 cases, its Sol judging $7). For big runs use the cheap model as judge "
          "(`--model judge=azure:gpt-6-luna`, Haiku or Flash-Lite): 20× cheaper on GPT-6. Keep "
          "the judge fixed across runs you compare, since the judge model is part of the suite id. "
          "Spot-check it against hand labels (`compare-runs judge-agreement`) before trusting it.\n")

    print("### Per step (openai profile, per case)\n")
    print("| Step | Split | Measured run | Strategy $/case | Judge (Sol) $/case | Duration | Judge requests |")
    print("|---|---|---|---|---|---|---|")
    for s in steps:
        print(f"| `{s.id}` | {s.split} | {s.run.name} | {money(s.cost(gpt))} | "
              f"{money(s.judge_cost('gpt-6-sol'))} | {s.run.duration:.1f} s | {s.run.jreq:.0f} |")


if __name__ == "__main__":
    main()
