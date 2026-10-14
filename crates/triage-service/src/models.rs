//! Chat and decision-model calls, addressed as `provider:model`.
//!
//! Every call records one counted span (`chat` or `decision`) and returns the
//! [`Usage`] it cost. Counted spans must only be created inside `ctx.run`, so
//! call these through [`journal`], never directly from a handler body: on a
//! replay Restate returns the journaled result without running the call, and
//! the span (and the bill) is recorded exactly once.
//!
//! A request carries the model *address* in its `model` field
//! (`ChatRequest::new("ollama:qwen3:4b")`); the call resolves it to a provider
//! and swaps in the provider's model id before sending.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::future::Future;
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

use allama::extract::Extractor;
use allama::prelude::*;
use allama_providers::ModelAddress;
use restate_sdk::prelude::*;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tracing::{Instrument, field::Empty, info_span, warn};
use triage_core::Usage;

use allama::DynDecisionProvider;
pub use allama::decision::Question;

pub use allama::{
    ChatRequest, DecisionRequest, Message, ReasoningEffort, ToolCall, ToolDefinition,
    ToolResultContent,
};

/// A failed model or tool call.
///
/// `terminal` errors (a bad address, a missing API key, a request the
/// provider rejects, output that can't be parsed) fail the same way on every
/// retry, so [`journal`] stops retrying them at once.
#[derive(Debug)]
pub struct CallError {
    message: String,
    terminal: bool,
}

impl CallError {
    pub fn terminal(message: impl Into<String>) -> Self {
        CallError {
            message: message.into(),
            terminal: true,
        }
    }

    pub fn retryable(message: impl Into<String>) -> Self {
        CallError {
            message: message.into(),
            terminal: false,
        }
    }

    pub fn is_terminal(&self) -> bool {
        self.terminal
    }
}

impl fmt::Display for CallError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for CallError {}

impl From<allama::Error> for CallError {
    fn from(e: allama::Error) -> Self {
        // `Display` carries the provider's message; `Debug` withholds it.
        CallError {
            message: e.to_string(),
            terminal: !e.is_retryable(),
        }
    }
}

/// Runs `call` inside `ctx.run` under `name`, so it executes (and is billed)
/// once: a retried or resumed invocation gets the journaled result back.
///
/// Retryable failures are retried a few times before the run fails with a
/// terminal error; terminal ones fail it at once. The provider clients retry
/// transient HTTP errors on their own too, so this only covers what's left.
///
/// ```ignore
/// let out: Structured<MyOutput> = journal(&ctx, "classify", models::extract("classify", request)).await?;
/// usage += &out.usage;
/// ```
pub async fn journal<T, F>(
    ctx: &WorkflowContext<'_>,
    name: &str,
    call: F,
) -> Result<T, TerminalError>
where
    F: Future<Output = Result<T, CallError>> + Send,
    T: Serialize + DeserializeOwned + 'static,
{
    let policy = RunRetryPolicy::default()
        .initial_delay(Duration::from_millis(500))
        .exponentiation_factor(2.0)
        .max_delay(Duration::from_secs(10))
        .max_attempts(3);
    let key = ctx.key().to_owned();
    let Json(value) = ctx
        .run(|| async move {
            if debug_fail_once(&key) {
                return Err(HandlerError::from(CallError::retryable(format!(
                    "{FAIL_ONCE_AFTER_RUNS}: failing this attempt on purpose"
                ))));
            }
            match call.await {
                Ok(value) => Ok(Json(value)),
                Err(e) if e.is_terminal() => Err(TerminalError::new(e.message).into()),
                Err(e) => Err(HandlerError::from(e)),
            }
        })
        .name(name)
        .retry_policy(policy)
        .await?;
    Ok(value)
}

/// Debug knob for checking replay live (logs, journaled calls). With
/// `FAIL_ONCE_AFTER_RUNS=N` set, the call after the first N journaled calls
/// this process executes for an invocation fails once with a retryable
/// error, before calling anything. Restate then retries the invocation: the
/// new attempt replays the N journaled calls and runs the rest live. Unset
/// (the default), it does nothing.
const FAIL_ONCE_AFTER_RUNS: &str = "FAIL_ONCE_AFTER_RUNS";

static FAIL_ONCE: LazyLock<Option<usize>> = LazyLock::new(|| {
    std::env::var(FAIL_ONCE_AFTER_RUNS)
        .ok()
        .and_then(|v| v.parse().ok())
});
/// Per invocation key: calls executed (not replayed), and whether it failed.
static FAIL_ONCE_STATE: LazyLock<Mutex<HashMap<String, (usize, bool)>>> =
    LazyLock::new(Default::default);

fn debug_fail_once(key: &str) -> bool {
    let Some(after) = *FAIL_ONCE else {
        return false;
    };
    let mut state = FAIL_ONCE_STATE.lock().expect("lock poisoned");
    let (executed, failed) = state.entry(key.to_owned()).or_default();
    if !*failed && *executed == after {
        *failed = true;
        warn!(key, after, "{FAIL_ONCE_AFTER_RUNS}: failing this attempt");
        return true;
    }
    *executed += 1;
    false
}

/// Parses a `provider:model` address, splitting at the first `:`
/// (`ollama:qwen3:4b` is provider `ollama`, model `qwen3:4b`).
pub fn parse_address(address: &str) -> Result<ModelAddress, CallError> {
    address
        .parse()
        .map_err(|e: allama::Error| CallError::terminal(e.to_string()))
}

/// Provider clients, built once per provider key: building one reads the
/// environment and sets up an HTTP client with its connection pool.
static CHAT_PROVIDERS: LazyLock<Mutex<HashMap<String, DynChatProvider>>> =
    LazyLock::new(Default::default);
static DECISION_PROVIDERS: LazyLock<Mutex<HashMap<String, DynDecisionProvider>>> =
    LazyLock::new(Default::default);

fn chat_provider(key: &str) -> Result<DynChatProvider, CallError> {
    let mut cache = CHAT_PROVIDERS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(provider) = cache.get(key) {
        return Ok(provider.clone());
    }
    let provider = allama_providers::chat::from_provider(key)?;
    cache.insert(key.to_owned(), provider.clone());
    Ok(provider)
}

fn decision_provider(key: &str) -> Result<DynDecisionProvider, CallError> {
    let mut cache = DECISION_PROVIDERS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(provider) = cache.get(key) {
        return Ok(provider.clone());
    }
    let provider = allama_providers::decision::from_provider(key)?;
    cache.insert(key.to_owned(), provider.clone());
    Ok(provider)
}

/// Resolves the request's address and applies per-provider defaults.
fn prepare_chat(request: &mut ChatRequest) -> Result<(ModelAddress, DynChatProvider), CallError> {
    let address = parse_address(&request.model)?;
    let provider = chat_provider(address.key())?;
    request.model = address.model().to_owned();
    default_reasoning(address.key(), request);
    Ok((address, provider))
}

/// Turns reasoning off where leaving it at the provider's default breaks or
/// wastes a call, unless the request set it.
fn default_reasoning(provider: &str, request: &mut ChatRequest) {
    if request.reasoning.is_some() {
        return;
    }
    // Qwen3 and other local reasoning models think by default, which burns
    // most of the tokens (and minutes on a CPU) of a classification call.
    // Ollama maps `reasoning_effort: none` to `think: false`.
    let local = provider == "ollama";
    // GPT-6 Sol and Luna on /chat/completions refuse function tools unless
    // reasoning_effort is none (only /responses allows both). Azure's
    // provider and `openai-chat` use /chat/completions; `openai` uses
    // /responses and keeps its reasoning.
    let chat_completions = matches!(provider, "azure" | "openai-chat");
    let chat_tools = chat_completions && !request.tools.is_empty();
    if local || chat_tools {
        request.reasoning = Some(ReasoningEffort::None.into());
    }
}

/// Checks `request` (addressed as `provider:model`) the way [`chat`] would
/// send it, defaults applied, without sending it: the provider encodes it
/// and refuses what it would refuse before a call. Credentials and the
/// model's existence are not checked.
pub fn validate(mut request: ChatRequest) -> Result<allama::ChatValidation, CallError> {
    let (_, provider) = prepare_chat(&mut request)?;
    Ok(provider.validate_chat_request(&request)?)
}

/// A value parsed from a model's structured output, with what it cost.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Structured<T> {
    pub value: T,
    pub usage: Usage,
}

/// One structured-output call: the model's answer parsed into `T`.
///
/// `step` names the call in telemetry (`llm.step`). The request must not
/// carry tools. Uses the provider's JSON-schema response format where it has
/// one (Ollama does), with local repair of near-miss JSON and one re-ask when
/// the output still doesn't parse; the re-ask is billed and counted.
pub async fn extract<T>(step: &str, mut request: ChatRequest) -> Result<Structured<T>, CallError>
where
    T: JsonSchema + DeserializeOwned + Send,
{
    let (address, provider) = prepare_chat(&mut request)?;
    let span = chat_span(&address, step);
    async move {
        let started = Instant::now();
        let result = Extractor::new()
            .repair(true)
            .retries(1)
            .extract::<_, T>(&provider, &request)
            .await;
        let latency = started.elapsed();
        match result {
            Ok(extracted) => {
                let usage = record_chat(
                    &address,
                    extracted.metadata.usage.as_ref(),
                    u64::from(extracted.metadata.passes.max(1)),
                    latency,
                    extracted.response.finish_reason.as_ref(),
                );
                Ok(Structured {
                    value: extracted.value,
                    usage,
                })
            }
            Err(e) => {
                let spent = match &e {
                    allama::Error::Extract { failure, .. } => failure.metadata().usage.clone(),
                    other => {
                        allama::extract::ExtractionMetadata::from_error(other).and_then(|m| m.usage)
                    }
                };
                record_chat(&address, spent.as_ref(), 1, latency, None);
                tracing::Span::current().record("error.type", e.kind_name());
                warn!(error = %e, "structured output call failed");
                Err(e.into())
            }
        }
    }
    .instrument(span)
    .await
}

/// One chat turn of a tool-using conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatTurn {
    /// The model's turn, to append to the conversation as it is (it keeps
    /// the tool calls and any provider state they need).
    pub message: Message,
    /// The text the model wrote, if any.
    pub text: Option<String>,
    /// The tools the model wants called, in order.
    pub tool_calls: Vec<ToolCall>,
    /// `stop`, `tool_calls`, `length`, ... as the provider reported it.
    pub finish_reason: Option<String>,
    pub usage: Usage,
}

/// One plain chat call, with whatever tools the request offers.
///
/// An agent-loop turn: send the request, journal the [`ChatTurn`], append
/// `turn.message` to `request.messages`, run each of `turn.tool_calls` and
/// add its result with `request.push_tool_result(&call, output)`, repeat
/// until the turn has no tool calls.
pub async fn chat(step: &str, mut request: ChatRequest) -> Result<ChatTurn, CallError> {
    let (address, provider) = prepare_chat(&mut request)?;
    let span = chat_span(&address, step);
    async move {
        let started = Instant::now();
        let result = provider.chat(&request).await;
        let latency = started.elapsed();
        match result {
            Ok(response) => {
                let usage = record_chat(
                    &address,
                    response.usage.as_ref(),
                    1,
                    latency,
                    response.finish_reason.as_ref(),
                );
                Ok(ChatTurn {
                    text: response.text(),
                    tool_calls: response.tool_calls().cloned().collect(),
                    finish_reason: response
                        .finish_reason
                        .as_ref()
                        .map(|r| r.as_str().to_owned()),
                    message: response.into_assistant_message(),
                    usage,
                })
            }
            Err(e) => {
                tracing::Span::current().record("error.type", e.kind_name());
                warn!(error = %e, "chat call failed");
                Err(e.into())
            }
        }
    }
    .instrument(span)
    .await
}

fn chat_span(address: &ModelAddress, step: &str) -> tracing::Span {
    info_span!(
        "chat",
        "gen_ai.operation.name" = "chat",
        "gen_ai.system" = address.key(),
        "gen_ai.request.model" = address.model(),
        "gen_ai.response.finish_reasons" = Empty,
        "gen_ai.usage.input_tokens" = Empty,
        "gen_ai.usage.output_tokens" = Empty,
        "llm.step" = step,
        "llm.calls" = Empty,
        "llm.latency_ms" = Empty,
        "llm.cost_usd" = Empty,
        "error.type" = Empty,
    )
}

/// Records usage on the current (call) span and converts it to the triage
/// [`Usage`]. Logfire exports unsigned integers as strings, so integer
/// attributes are recorded as `i64` to stay numbers in SQL.
fn record_chat(
    address: &ModelAddress,
    usage: Option<&allama::Usage>,
    calls: u64,
    latency: Duration,
    finish_reason: Option<&allama::FinishReason>,
) -> Usage {
    let span = tracing::Span::current();
    let out = usage_of(address, usage, latency);
    let out = Usage {
        llm_calls: calls,
        ..out
    };
    span.record("gen_ai.usage.input_tokens", out.input_tokens as i64);
    span.record("gen_ai.usage.output_tokens", out.output_tokens as i64);
    span.record("llm.calls", calls as i64);
    span.record("llm.latency_ms", out.latency_ms as i64);
    span.record("llm.cost_usd", out.cost_usd);
    if let Some(reason) = finish_reason {
        span.record("gen_ai.response.finish_reasons", reason.as_str());
    }
    out
}

fn usage_of(address: &ModelAddress, usage: Option<&allama::Usage>, latency: Duration) -> Usage {
    let input_tokens = usage.and_then(|u| u.input_tokens).unwrap_or(0);
    let output_tokens = usage.and_then(|u| u.output_tokens).unwrap_or(0);
    Usage {
        input_tokens,
        output_tokens,
        latency_ms: latency.as_millis() as u64,
        cost_usd: cost(&address.to_string(), input_tokens, output_tokens),
        ..Usage::default()
    }
}

fn cost(address: &str, input_tokens: u64, output_tokens: u64) -> f64 {
    match triage_core::price(address) {
        Some(price) => price.cost(input_tokens, output_tokens),
        None => {
            warn!(
                model = address,
                "no price for model, counting its cost as 0"
            );
            0.0
        }
    }
}

/// A decision-model answer to one question, flattened for journaling and for
/// routing on confidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DecisionAnswer {
    YesNo {
        /// The probability of yes.
        probability: f64,
    },
    Choice {
        chosen: String,
        /// The provider's own confidence statistic, when it reports one. Not
        /// comparable across providers (see `allama_systemone::Confidence`).
        confidence: Option<f64>,
        probabilities: BTreeMap<String, f64>,
    },
    Score {
        /// The expected level, 0-based, as a real number.
        score: f64,
        /// The most likely level, 0-based.
        most_likely: Option<usize>,
        confidence: Option<f64>,
        /// One probability per level, in level order.
        probabilities: Vec<f64>,
    },
}

impl DecisionAnswer {
    /// The probability of the answer given: `max(p, 1 - p)` for yes/no, the
    /// chosen option's for a choice, the most likely level's for a score.
    pub fn top_probability(&self) -> f64 {
        match self {
            DecisionAnswer::YesNo { probability } => probability.max(1.0 - probability),
            DecisionAnswer::Choice {
                chosen,
                probabilities,
                ..
            } => probabilities.get(chosen).copied().unwrap_or(0.0),
            DecisionAnswer::Score {
                most_likely,
                probabilities,
                ..
            } => most_likely
                .and_then(|i| probabilities.get(i).copied())
                .unwrap_or(0.0),
        }
    }
}

/// The answers of one decision call, by question name.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionSummary {
    pub answers: BTreeMap<String, DecisionAnswer>,
    pub usage: Usage,
}

impl DecisionSummary {
    pub fn yes_no(&self, name: &str) -> Option<f64> {
        match self.answers.get(name)? {
            DecisionAnswer::YesNo { probability } => Some(*probability),
            _ => None,
        }
    }

    /// The chosen option and its probability.
    pub fn choice(&self, name: &str) -> Option<(&str, f64)> {
        match self.answers.get(name)? {
            a @ DecisionAnswer::Choice { chosen, .. } => Some((chosen, a.top_probability())),
            _ => None,
        }
    }
}

/// One decision-model call. `request.model` is the address
/// (`ollama:tev1:0.8b-q8_0`, `typesafe:jev-latest`,
/// `cloudflare:@cf/cloudflare/clef`). On System One every question needs
/// instructions and every choice option a description.
pub async fn decide(
    step: &str,
    mut request: DecisionRequest,
) -> Result<DecisionSummary, CallError> {
    let address = parse_address(&request.model)?;
    let provider = decision_provider(address.key())?;
    request.model = address.model().to_owned();
    let span = info_span!(
        "decision",
        "gen_ai.operation.name" = "decision",
        "gen_ai.system" = address.key(),
        "gen_ai.request.model" = address.model(),
        "gen_ai.usage.input_tokens" = Empty,
        "gen_ai.usage.output_tokens" = Empty,
        "llm.step" = step,
        "llm.questions" = request.questions.len() as i64,
        "llm.latency_ms" = Empty,
        "llm.cost_usd" = Empty,
        "error.type" = Empty,
    );
    async move {
        let started = Instant::now();
        let result = provider.decide(&request).await;
        let latency = started.elapsed();
        let span = tracing::Span::current();
        let response = match result {
            Ok(response) => response,
            Err(e) => {
                span.record("error.type", e.kind_name());
                warn!(error = %e, "decision call failed");
                return Err(e.into());
            }
        };
        let usage = Usage {
            decision_calls: 1,
            ..usage_of(&address, response.usage.as_ref(), latency)
        };
        span.record("gen_ai.usage.input_tokens", usage.input_tokens as i64);
        span.record("gen_ai.usage.output_tokens", usage.output_tokens as i64);
        span.record("llm.latency_ms", usage.latency_ms as i64);
        span.record("llm.cost_usd", usage.cost_usd);

        let confidence = response
            .provider_data
            .get::<allama_systemone::Confidence>()
            .ok()
            .flatten()
            .map(|c| c.0)
            .unwrap_or_default();
        let answers = response
            .answers
            .iter()
            .map(|(name, answer)| {
                let confidence = confidence.get(name).copied();
                (name.clone(), summarize(answer, confidence))
            })
            .collect();
        Ok(DecisionSummary { answers, usage })
    }
    .instrument(span)
    .await
}

fn summarize(answer: &allama::decision::Answer, confidence: Option<f64>) -> DecisionAnswer {
    use allama::decision::Answer;
    match answer {
        Answer::YesNo(a) => DecisionAnswer::YesNo {
            probability: a.probability,
        },
        Answer::Choice(a) => DecisionAnswer::Choice {
            chosen: a.chosen.clone(),
            confidence,
            probabilities: a.probabilities.iter().cloned().collect(),
        },
        Answer::Score(a) => DecisionAnswer::Score {
            score: a.score,
            most_likely: a.most_likely(),
            confidence,
            probabilities: a.probabilities.clone(),
        },
        // `Answer` is non-exhaustive; a kind added later reads as an
        // undecided yes/no rather than failing the triage.
        #[allow(unreachable_patterns)]
        _ => DecisionAnswer::YesNo { probability: 0.5 },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn address_splits_at_first_colon() {
        let a = parse_address("ollama:qwen3:4b").unwrap();
        assert_eq!((a.key(), a.model()), ("ollama", "qwen3:4b"));
        assert!(parse_address("qwen3").is_err());
    }

    #[test]
    fn unknown_provider_is_terminal() {
        let mut request = ChatRequest::new("nosuchprovider:model");
        let err = prepare_chat(&mut request).err().unwrap();
        assert!(err.is_terminal(), "{err}");
    }

    #[test]
    fn ollama_thinking_is_off_by_default() {
        let mut request = ChatRequest::new("ollama:qwen3:1.7b");
        prepare_chat(&mut request).unwrap();
        assert_eq!(request.model, "qwen3:1.7b");
        assert_eq!(
            request.reasoning.and_then(|r| r.effort),
            Some(ReasoningEffort::None)
        );
    }

    #[test]
    fn azure_reasoning_is_off_only_with_tools() {
        let effort = |r: &ChatRequest| r.reasoning.as_ref().and_then(|r| r.effort);

        let mut plain = ChatRequest::new("gpt-6-luna");
        default_reasoning("azure", &mut plain);
        assert_eq!(effort(&plain), None);

        let tool = crate::tools::definitions().remove(0);
        let mut with_tools = ChatRequest::new("gpt-6-luna").tools(vec![tool]);
        default_reasoning("azure", &mut with_tools);
        assert_eq!(effort(&with_tools), Some(ReasoningEffort::None));

        let mut set = ChatRequest::new("gpt-6-luna")
            .tools(crate::tools::definitions())
            .reasoning(ReasoningEffort::Low);
        default_reasoning("azure", &mut set);
        assert_eq!(effort(&set), Some(ReasoningEffort::Low));
    }

    #[test]
    fn openai_reasoning_stays_on_responses_and_goes_off_on_chat_completions() {
        let effort = |r: &ChatRequest| r.reasoning.as_ref().and_then(|r| r.effort);
        let with_tools = || ChatRequest::new("gpt-6-luna").tools(crate::tools::definitions());

        let mut responses = with_tools();
        default_reasoning("openai", &mut responses);
        assert_eq!(effort(&responses), None);

        let mut chat_completions = with_tools();
        default_reasoning("openai-chat", &mut chat_completions);
        assert_eq!(effort(&chat_completions), Some(ReasoningEffort::None));
    }

    #[test]
    fn top_probability() {
        let a = DecisionAnswer::Choice {
            chosen: "bug".into(),
            confidence: None,
            probabilities: [("bug".into(), 0.7), ("feature".into(), 0.3)].into(),
        };
        assert_eq!(a.top_probability(), 0.7);
        assert_eq!(
            DecisionAnswer::YesNo { probability: 0.2 }.top_probability(),
            0.8
        );
    }
}
