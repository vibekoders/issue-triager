"""LLM judges for the two fields that have no labels: `quality` and `suggested_reply`.

Both only run when the strategy produced the field (otherwise they return
nothing, like the code evaluators), and only when `--judge-model` is set.

- `QualityJudge` rates the issue itself on triage-core's `QUALITY_RUBRIC`
  (the scale the strategies were prompted with) without seeing the
  strategy's rating, so it can't anchor on it. The score is how far apart the
  two ratings are.
- `ReplyJudge` is pydantic-evals' `LLMJudge` with a reply rubric; it sees the
  issue and the triage the reply was written from.

The judge model is wrapped in `RecordingModel`, which adds every response's
token usage to the `JudgeLedger` entry of the case being judged. That is how
the judge's tokens and cost are reported separately: pydantic-evals only
puts task-span usage into case metrics, and `LLMJudge` doesn't return usage.

Models are addressed like the Rust service's (`ollama:`, `azure:`, `openai:`,
`anthropic:`, `gemini:`; see `make_judge_model`); any other prefix is passed
to pydantic-ai as a model string. `ollama:<model>` uses `OLLAMA_BASE_URL`
(default here: http://localhost:11434/v1).
"""

from __future__ import annotations

import os
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models import Model, ModelRequestParameters, infer_model
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext, LLMJudge
from pydantic_evals.evaluators.common import OutputConfig

from .evaluators import EVALUATORS_VERSION, as_dict, triage_of
from .suite import RUBRIC_VERSION, price_cost, rubric_text

# Qwen, like the strategies, against PLAN.md's "different family" rule: the
# local Ollama has only Qwen-based models. Pass another family's model for
# judged results that matter (see "The judge's model family" in the README).
DEFAULT_JUDGE_MODEL = "ollama:qwen3:4b"
BODY_LIMIT = 6000

REPLY_RUBRIC = """\
<Output> is the first reply a maintainer would post on the GitHub issue in <Input>, \
drafted from the triage in <Input>. It passes when all of these hold:
1. If the issue lacks information needed to act on it, the reply asks for exactly that \
information (no more, no less); if nothing is missing, it doesn't ask for anything.
2. It is polite and professional, and speaks to the author.
3. It states no facts that are not in the issue (no invented versions, causes, fixes or links).
4. It fits the kind of issue (bug, feature, question, docs, test failure, flake).
Set score between 0 and 1: the share of the four points the reply meets fully."""


@dataclass
class JudgeUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0
    cost_usd: float = 0.0
    priced: bool = True

    def add(self, other: JudgeUsage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.requests += other.requests
        self.cost_usd += other.cost_usd
        self.priced = self.priced and other.priced


_current: ContextVar[JudgeUsage | None] = ContextVar("judge_usage", default=None)


@dataclass(init=False)
class RecordingModel(WrapperModel):
    """Adds each response's usage to the `JudgeUsage` of the current case."""

    address: str

    def __init__(self, address: str, wrapped: Model | None = None):
        super().__init__(wrapped or infer_model(address))
        self.address = address

    async def request(
        self,
        messages: list[Any],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> Any:
        response = await super().request(messages, model_settings, model_request_parameters)
        usage = _current.get()
        if usage is not None:
            inp, out = response.usage.input_tokens or 0, response.usage.output_tokens or 0
            cost = price_cost(self.address, inp, out)
            usage.add(JudgeUsage(inp, out, 1, cost or 0.0, cost is not None))
        return response


@dataclass
class JudgeLedger:
    """Judge usage per workflow key (one entry per case and repeat)."""

    by_key: dict[str, JudgeUsage] = field(default_factory=dict)

    def entry(self, key: str) -> JudgeUsage:
        return self.by_key.setdefault(key, JudgeUsage())

    def total(self) -> JudgeUsage:
        total = JudgeUsage()
        for u in self.by_key.values():
            total.add(u)
        return total


# Providers whose models (GPT-6) reason: they ignore `temperature`, with a
# warning on every call, so it isn't sent to them.
_REASONING_PROVIDERS = ("azure:", "openai:")


def judge_model_settings(address: str) -> ModelSettings:
    """Deterministic where the model allows it, and with qwen3's thinking turned off on Ollama (slow on CPU).

    No `temperature` for GPT-6 (`azure:`, `openai:`), which reasons and ignores
    it, nor for Gemini (`gemini:`): Google recommends leaving Gemini 3 at its
    default of 1.0, as lower values can make it loop.
    """
    settings: dict[str, Any] = {}
    if not address.startswith((*_REASONING_PROVIDERS, "gemini:")):
        settings["temperature"] = 0.0
    if address.startswith("ollama:"):
        settings["openai_reasoning_effort"] = "none"
    return settings  # type: ignore[return-value]


def make_judge_model(address: str) -> RecordingModel:
    """The judge model, recording usage.

    Addresses are the Rust service's (`provider:model`), so a profile names a
    model the same way for the strategies and the judge:

    - `ollama:` - the local Ollama. Structured output goes through
      `response_format` (native mode) instead of a tool call: with a tool call
      qwen3 keeps thinking even when asked not to, which costs thousands of
      tokens and minutes per case on CPU.
    - `azure:<deployment>` - see `azure_model`.
    - `openai:<model>` - OpenAI's Responses API (`OPENAI_API_KEY`), for the same
      reason as Azure: on `/chat/completions` GPT-6 refuses function tools while
      it reasons.
    - `anthropic:<model>` - `ANTHROPIC_API_KEY`.
    - `gemini:<model>` - the Gemini API through pydantic-ai's Google provider,
      with the key the Rust service reads: `GEMINI_API_KEY`, else
      `GOOGLE_API_KEY`.

    Anything else goes to pydantic-ai's `infer_model` as it is (`google:...`,
    `groq:...`, ...).
    """
    provider, _, name = address.partition(":")
    if provider == "ollama":
        from pydantic_ai.models.ollama import OllamaModel
        from pydantic_ai.providers.ollama import OllamaProvider

        os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        model: Model = OllamaModel(
            name,
            provider=OllamaProvider(),
            profile={"default_structured_output_mode": "native"},
        )
    elif provider == "azure":
        model = azure_model(name)
    elif provider == "openai":
        from pydantic_ai.models.openai import OpenAIResponsesModel
        from pydantic_ai.providers.openai import OpenAIProvider

        model = OpenAIResponsesModel(name, provider=OpenAIProvider())
    elif provider == "anthropic":
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        model = AnthropicModel(name, provider=AnthropicProvider())
    elif provider == "gemini":
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        model = GoogleModel(name, provider=GoogleProvider(api_key=api_key))
    else:
        return RecordingModel(address)
    return RecordingModel(address, model)


def azure_model(deployment: str) -> Model:
    """An Azure OpenAI deployment through the v1 API, as the Rust service reaches it.

    Configured like the service: `AZURE_OPENAI_RESOURCE` (or
    `AZURE_OPENAI_ENDPOINT`) and `AZURE_OPENAI_API_KEY`; the v1 API needs no
    API version. It uses the Responses API because on `/chat/completions`
    GPT-6 deployments refuse function tools (pydantic-ai's structured output)
    unless reasoning is turned off.
    """
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT") or (
        f"https://{os.environ['AZURE_OPENAI_RESOURCE']}.openai.azure.com"
    )
    base_url = endpoint.rstrip("/")
    if not base_url.endswith("/openai/v1"):
        base_url += "/openai/v1"
    provider = OpenAIProvider(base_url=base_url, api_key=os.environ["AZURE_OPENAI_API_KEY"])
    return OpenAIResponsesModel(deployment, provider=provider)


def issue_text(inputs: Any) -> str:
    issue = as_dict(inputs).get("issue", {})
    body = issue.get("body") or ""
    if len(body) > BODY_LIMIT:
        body = body[:BODY_LIMIT] + "\n[... cut ...]"
    return f"Title: {issue.get('title', '')}\n\n{body}"


class QualityRating(BaseModel):
    reason: str = Field(description="Why, in at most 30 words.")
    rating: int = Field(ge=1, le=5, description="The rubric level that fits the issue best.")


QUALITY_INSTRUCTIONS = f"""\
You rate GitHub issue reports filed against kubernetes/kubernetes, as a maintainer \
triaging them would. Rate the issue as it was filed, on this rubric:

{rubric_text()}

Judge only the report itself: is the problem or request clear, and does it carry what \
a maintainer needs to act on it? Not whether you agree with it."""


class _JudgeSpec:
    """Mixin (pydantic-evals' Evaluator base can't be subclassed abstractly)."""

    model: RecordingModel

    def build_serialization_arguments(self) -> dict[str, Any]:
        """The spec shows the model address, not the model object and the ledger."""
        return {"model": self.model.address}


@dataclass
class QualityJudge(_JudgeSpec, Evaluator[Any, Any, Any]):
    """Rates the issue on the quality rubric and compares with the strategy's rating."""

    model: RecordingModel
    ledger: JudgeLedger

    def get_evaluator_version(self) -> str | None:
        return f"{EVALUATORS_VERSION}/rubric-{RUBRIC_VERSION}"

    async def evaluate(self, ctx: EvaluatorContext[Any, Any, Any]) -> dict[str, Any]:
        field_ = triage_of(ctx.output).get("quality")
        if not field_:
            return {}
        predicted = int(field_["value"])
        agent = Agent(
            self.model,
            output_type=QualityRating,
            instructions=QUALITY_INSTRUCTIONS,
            name="quality_judge",
        )
        token = _current.set(self.ledger.entry(ctx.attributes.get("workflow_key", ctx.name or "")))
        try:
            result = await agent.run(issue_text(ctx.inputs), model_settings=judge_model_settings(self.model.address))
        finally:
            _current.reset(token)
        judged = result.output.rating
        error = abs(predicted - judged)
        return {
            "quality": EvaluationReason(value=1 - error / 4, reason=f"judge {judged}: {result.output.reason}"),
            "quality_abs_error": error,
            "quality_agree": error == 0,
            "quality_judge": judged,
        }


@dataclass
class ReplyJudge(_JudgeSpec, Evaluator[Any, Any, Any]):
    """`LLMJudge` on `suggested_reply`, given the issue and the triage it was written from."""

    model: RecordingModel
    ledger: JudgeLedger

    def get_evaluator_version(self) -> str | None:
        return f"{EVALUATORS_VERSION}/reply-rubric-{RUBRIC_VERSION}"

    async def evaluate(self, ctx: EvaluatorContext[Any, Any, Any]) -> dict[str, Any]:
        triage = triage_of(ctx.output)
        reply = (triage.get("suggested_reply") or {}).get("value")
        if not reply:
            return {}
        repro = (triage.get("reproducibility") or {}).get("value") or {}
        judge_inputs = {
            "issue": issue_text(ctx.inputs),
            "triage": {
                "kind": (triage.get("kind") or {}).get("value"),
                "missing_information": repro.get("missing"),
                "enough_info": repro.get("enough_info"),
            },
        }
        judge = LLMJudge(
            rubric=REPLY_RUBRIC,
            model=self.model,
            include_input=True,
            model_settings=judge_model_settings(self.model.address),
            score=OutputConfig(evaluation_name="reply", include_reason=False),
            assertion=OutputConfig(evaluation_name="reply_pass", include_reason=True),
        )
        token = _current.set(self.ledger.entry(ctx.attributes.get("workflow_key", ctx.name or "")))
        try:
            return dict(await judge.evaluate(replace(ctx, inputs=judge_inputs, output=reply)))
        finally:
            _current.reset(token)


def judge_evaluators(address: str, ledger: JudgeLedger) -> list[Evaluator[Any, Any, Any]]:
    model = make_judge_model(address)
    return [QualityJudge(model, ledger), ReplyJudge(model, ledger)]
