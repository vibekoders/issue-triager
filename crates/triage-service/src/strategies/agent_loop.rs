//! A durable tool-using agent loop, shared by `Agent`, `DraftVerify` and the
//! `Workflow` sub-agent.
//!
//! Each chat turn and each tool call runs in its own `ctx.run` (through
//! [`journal`]), so a retried or resumed invocation replays the journaled
//! turns instead of calling the model again. After the loop, one
//! structured-output call turns what the agent found into the caller's
//! answer type `T`:
//!
//! ```ignore
//! let spec = AgentSpec::new("agent", &model, system, prompt)
//!     .tools(tools::definitions())
//!     .max_turns(4);
//! let finish = Finish::new(final_system, final_context, final_instructions);
//! let out: AgentOutcome<MyAnswer> = agent_loop::run(&ctx, &tool_ctx, &spec, &finish).await?;
//! ```
//!
//! The final call sees the findings as plain text, not as the tool-call
//! conversation: it carries no tools (structured output refuses them, and
//! Anthropic rejects tool calls in a request without tool definitions), and
//! a condensed transcript fits a small local model's context better.
//!
//! With no tools offered there is no loop: the final call is the only one.
//!
//! The loop is forgiving with small models: malformed tool arguments and
//! unknown tool names become error results the model can react to, and
//! hitting `max_turns` still ends with the final answer.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

use allama::decision::DecisionInput;
use restate_sdk::prelude::*;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tracing::info;
use triage_core::Usage;

use super::decision_triage::truncate;
use crate::models::{
    self, CallError, ChatRequest, ChatTurn, DecisionAnswer, DecisionRequest, Question, Structured,
    ToolCall, ToolDefinition, journal,
};
use crate::tools::{self, ToolContext, ToolOutput};

/// The name of the optional decision-model tool.
pub const DECIDE: &str = "decide";

/// Tool results are cut to this many characters in the conversation, so a
/// few of them fit a small model's context window next to the issue.
pub const MAX_TOOL_OUTPUT_CHARS: usize = 2500;
/// Each tool result is cut shorter still in the final call's transcript.
const MAX_FINDING_CHARS: usize = 1500;
/// Reference lookups (labels, templates, releases) are long and mostly
/// restate what the final prompt already says, so the transcript keeps only
/// their start.
const MAX_REFERENCE_FINDING_CHARS: usize = 400;
/// Tool calls beyond this many in one turn get an error result instead of
/// running: small models sometimes fire the same search ten times.
const MAX_TOOL_CALLS_PER_TURN: usize = 4;
/// Output caps are ceilings, not budgets: reasoning models (GPT-6 on Azure)
/// spend part of them on reasoning before the answer, and a structured answer
/// cut off at the cap fails to parse. Local models stop well below them.
const TURN_OUTPUT_TOKENS: u32 = 2000;
const FINAL_OUTPUT_TOKENS: u32 = 6000;

/// What the agent is asked to do and with what.
#[derive(Debug, Clone)]
pub struct AgentSpec {
    /// Names the loop's journal entries (`{step}-turn-0`, `{step}-tool-0-1-...`,
    /// `{step}-final`) and its calls' `llm.step`. Must be unique among the
    /// loops of one workflow invocation.
    pub step: String,
    /// The chat model's address, e.g. `ollama:qwen3:4b`.
    pub model: String,
    pub system: String,
    /// The first user message: the task and the issue.
    pub prompt: String,
    /// The corpus tools to offer (see [`tools::definitions_for`]).
    pub tools: Vec<ToolDefinition>,
    /// Adds the [`DECIDE`] tool.
    pub decision_tool: Option<DecisionTool>,
    /// Chat turns before the loop stops and the final answer is taken.
    pub max_turns: u32,
    pub temperature: Option<f64>,
}

impl AgentSpec {
    pub fn new(
        step: impl Into<String>,
        model: impl Into<String>,
        system: impl Into<String>,
        prompt: impl Into<String>,
    ) -> Self {
        AgentSpec {
            step: step.into(),
            model: model.into(),
            system: system.into(),
            prompt: prompt.into(),
            tools: Vec::new(),
            decision_tool: None,
            max_turns: 8,
            temperature: None,
        }
    }

    pub fn tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = tools;
        self
    }

    pub fn decision_tool(mut self, tool: Option<DecisionTool>) -> Self {
        self.decision_tool = tool;
        self
    }

    pub fn max_turns(mut self, max_turns: u32) -> Self {
        self.max_turns = max_turns;
        self
    }

    pub fn temperature(mut self, temperature: Option<f64>) -> Self {
        self.temperature = temperature;
        self
    }

    /// The tool definitions offered to the model, the decide tool included.
    pub fn offered_tools(&self) -> Vec<ToolDefinition> {
        let mut defs = self.tools.clone();
        if self.decision_tool.is_some() {
            defs.push(decide_definition());
        }
        defs
    }
}

/// The `decide` tool: a decision model answering the agent's own yes/no or
/// choice question over `input` (usually [`super::decision_triage::input`]).
#[derive(Debug, Clone)]
pub struct DecisionTool {
    /// The decision model's address, e.g. `ollama:tev1:0.8b-q8_0`.
    pub model: String,
    pub input: DecisionInput,
}

/// How the final structured answer is asked for.
#[derive(Debug, Clone)]
pub struct Finish {
    pub system: String,
    /// What comes before the findings: the issue and what to fill in.
    pub context: String,
    /// What comes after them: the last instructions.
    pub instructions: String,
}

impl Finish {
    pub fn new(
        system: impl Into<String>,
        context: impl Into<String>,
        instructions: impl Into<String>,
    ) -> Self {
        Finish {
            system: system.into(),
            context: context.into(),
            instructions: instructions.into(),
        }
    }
}

/// What the loop produced.
#[derive(Debug, Clone)]
pub struct AgentOutcome<T> {
    /// The final structured answer.
    pub answer: T,
    /// Every turn, tool call and the final call, summed.
    pub usage: Usage,
    /// Chat turns taken (the final call not included).
    pub turns: u32,
    /// The loop stopped at `max_turns` with tool calls still coming.
    pub hit_max_turns: bool,
    /// Every tool call, in order.
    pub tool_calls: Vec<ToolRecord>,
    /// Issue numbers the corpus tools returned, first seen first. Answers
    /// that cite issues (duplicates) should be checked against these.
    pub found_issues: Vec<u64>,
}

impl<T> AgentOutcome<T> {
    /// Whether a corpus search or issue lookup succeeded at least once.
    pub fn searched(&self) -> bool {
        self.tool_calls
            .iter()
            .any(|c| !c.is_error && (c.name == tools::SEARCH_ISSUES || c.name == tools::GET_ISSUE))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolRecord {
    pub name: String,
    pub arguments: String,
    pub is_error: bool,
}

/// Runs the loop, then the final extraction into `T`.
///
/// Fails only when a model call fails for good (after the journal's
/// retries) or the final answer can't be parsed; tool problems are fed back
/// to the model. A caller that carries on after a failure should use
/// [`run_metered`], so the calls that did succeed are still counted.
pub async fn run<T>(
    ctx: &WorkflowContext<'_>,
    tool_ctx: &ToolContext,
    spec: &AgentSpec,
    finish: &Finish,
) -> Result<AgentOutcome<T>, TerminalError>
where
    T: JsonSchema + Serialize + DeserializeOwned + Send + 'static,
{
    run_metered(ctx, tool_ctx, spec, finish, &mut Usage::default()).await
}

/// [`run`], adding every journaled call's usage to `spent` as it goes, so
/// that on a failure `spent` still holds what the turns and tool calls before
/// it cost (they ran and were paid for; a replay takes the same path). On
/// success the outcome's `usage` is what this call added to `spent`.
pub async fn run_metered<T>(
    ctx: &WorkflowContext<'_>,
    tool_ctx: &ToolContext,
    spec: &AgentSpec,
    finish: &Finish,
    spent: &mut Usage,
) -> Result<AgentOutcome<T>, TerminalError>
where
    T: JsonSchema + Serialize + DeserializeOwned + Send + 'static,
{
    let step = spec.step.as_str();
    let offered = spec.offered_tools();
    let offered_names: Vec<String> = offered.iter().map(|d| d.name.clone()).collect();
    let mut request = ChatRequest::new(&spec.model)
        .system(spec.system.as_str())
        .user(spec.prompt.as_str())
        .max_output_tokens(TURN_OUTPUT_TOKENS)
        .tools(offered);
    if let Some(t) = spec.temperature {
        request = request.temperature(t);
    }

    let mut usage = Usage::default();
    let mut findings: Vec<Finding> = Vec::new();
    let mut tool_calls: Vec<ToolRecord> = Vec::new();
    let mut found = FoundIssues::default();
    let mut turns = 0;
    let mut hit_max_turns = true;
    // With nothing to call, a turn could only produce notes: skip the loop
    // and go straight to the final call, which sees the issue.
    let max_turns = if offered_names.is_empty() {
        hit_max_turns = false;
        0
    } else {
        spec.max_turns
    };
    for turn in 0..max_turns {
        let reply: ChatTurn = journal(
            ctx,
            &format!("{step}-turn-{turn}"),
            models::chat(step, request.clone()),
        )
        .await?;
        usage += &reply.usage;
        *spent += &reply.usage;
        turns += 1;
        if let Some(text) = reply
            .text
            .as_deref()
            .map(str::trim)
            .filter(|t| !t.is_empty())
        {
            findings.push(Finding::Note(text.to_owned()));
        }
        request.messages.push(reply.message.clone());
        if reply.tool_calls.is_empty() {
            hit_max_turns = false;
            break;
        }
        for (i, call) in reply.tool_calls.iter().enumerate() {
            let output = if i >= MAX_TOOL_CALLS_PER_TURN {
                error_output(format!(
                    "too many tool calls in one turn; make at most {MAX_TOOL_CALLS_PER_TURN} at a time"
                ))
            } else if !offered_names.contains(&call.name) {
                // Nothing runs, so there is nothing to journal: the same
                // turn always gets the same message.
                error_output(format!(
                    "unknown tool `{}`; available tools: {}",
                    call.name,
                    offered_names.join(", ")
                ))
            } else {
                let name = format!("{step}-tool-{turn}-{i}-{}", call.name);
                run_tool(ctx, &name, tool_ctx, spec.decision_tool.as_ref(), call).await?
            };
            usage += &output.usage;
            *spent += &output.usage;
            if !output.is_error {
                found.collect(&call.name, &output.content);
            }
            let shown = render(&call.name, &output.content);
            request.push_tool_result(call, truncate(&shown, MAX_TOOL_OUTPUT_CHARS));
            findings.push(Finding::Tool {
                name: call.name.clone(),
                arguments: call.arguments.clone(),
                result: truncate(&shown, finding_limit(&call.name)),
            });
            tool_calls.push(ToolRecord {
                name: call.name.clone(),
                arguments: call.arguments.clone(),
                is_error: output.is_error,
            });
        }
    }

    let mut final_request = ChatRequest::new(&spec.model)
        .system(finish.system.as_str())
        .user(final_prompt(finish, &findings))
        .max_output_tokens(FINAL_OUTPUT_TOKENS);
    if let Some(t) = spec.temperature {
        final_request = final_request.temperature(t);
    }
    let final_step = format!("{step}-final");
    let answer: Structured<T> = journal(
        ctx,
        &final_step,
        models::extract(&final_step, final_request),
    )
    .await?;
    usage += &answer.usage;
    *spent += &answer.usage;
    info!(
        step,
        turns,
        hit_max_turns,
        tool_calls = tool_calls.len() as i64,
        "agent loop done"
    );
    Ok(AgentOutcome {
        answer: answer.value,
        usage,
        turns,
        hit_max_turns,
        tool_calls,
        found_issues: found.numbers,
    })
}

async fn run_tool(
    ctx: &WorkflowContext<'_>,
    name: &str,
    tool_ctx: &ToolContext,
    decision_tool: Option<&DecisionTool>,
    call: &ToolCall,
) -> Result<ToolOutput, TerminalError> {
    let arguments = call.arguments.clone();
    if call.name == DECIDE {
        let tool = decision_tool
            .cloned()
            .expect("decide is only offered with a decision tool");
        journal(ctx, name, async move { decide(&tool, &arguments).await }).await
    } else {
        let tool_ctx = tool_ctx.clone();
        let tool_name = call.name.clone();
        journal(ctx, name, async move {
            tools::call(&tool_ctx, &tool_name, &arguments).await
        })
        .await
    }
}

fn error_output(message: String) -> ToolOutput {
    ToolOutput {
        content: format!("error: {message}"),
        is_error: true,
        usage: Usage::default(),
    }
}

/// Something the agent learned, for the final call's transcript.
enum Finding {
    Note(String),
    Tool {
        name: String,
        arguments: String,
        result: String,
    },
}

fn finding_limit(tool: &str) -> usize {
    match tool {
        tools::LIST_LABELS | tools::GET_ISSUE_TEMPLATES | tools::GET_RELEASES => {
            MAX_REFERENCE_FINDING_CHARS
        }
        _ => MAX_FINDING_CHARS,
    }
}

fn final_prompt(finish: &Finish, findings: &[Finding]) -> String {
    let mut p = String::new();
    p.push_str(finish.context.trim_end());
    p.push_str("\n\n## Findings from the investigation\n\n");
    if findings.is_empty() {
        p.push_str("(none: no tools were used)\n");
    }
    for finding in findings {
        match finding {
            Finding::Note(text) => {
                let _ = writeln!(p, "Note: {}\n", truncate(text, MAX_FINDING_CHARS));
            }
            Finding::Tool {
                name,
                arguments,
                result,
            } => {
                let _ = writeln!(p, "Called {name} {}:\n{result}\n", arguments.trim());
            }
        }
    }
    p.push('\n');
    p.push_str(finish.instructions.trim());
    p
}

/// Search results and issues as lines rather than JSON: the same content
/// in about half the tokens. Other results pass through.
fn render(name: &str, content: &str) -> String {
    let Ok(value) = serde_json::from_str::<Value>(content) else {
        return content.to_owned();
    };
    match (name, &value) {
        (tools::SEARCH_ISSUES, Value::Array(rows)) => {
            if rows.is_empty() {
                return "no matching issues".to_owned();
            }
            let mut out = String::new();
            for row in rows {
                let _ = writeln!(
                    out,
                    "#{} {} ({}): {}",
                    row["number"],
                    row["title"].as_str().unwrap_or_default(),
                    row["created_at"]
                        .as_str()
                        .map(|d| d.get(..10).unwrap_or(d))
                        .unwrap_or_default(),
                    row["snippet"]
                        .as_str()
                        .unwrap_or_default()
                        .split_whitespace()
                        .collect::<Vec<_>>()
                        .join(" "),
                );
            }
            out
        }
        (tools::GET_ISSUE, Value::Object(issue)) => format!(
            "#{} {} ({})\n{}",
            issue.get("number").cloned().unwrap_or(Value::Null),
            issue
                .get("title")
                .and_then(Value::as_str)
                .unwrap_or_default(),
            issue
                .get("created_at")
                .and_then(Value::as_str)
                .unwrap_or_default(),
            issue
                .get("body")
                .and_then(Value::as_str)
                .unwrap_or_default(),
        ),
        _ => content.to_owned(),
    }
}

/// Issue numbers in corpus tool results, in the order first seen.
#[derive(Default)]
struct FoundIssues {
    numbers: Vec<u64>,
    seen: BTreeSet<u64>,
}

impl FoundIssues {
    fn collect(&mut self, tool: &str, content: &str) {
        if tool != tools::SEARCH_ISSUES && tool != tools::GET_ISSUE {
            return;
        }
        let Ok(value) = serde_json::from_str::<Value>(content) else {
            return;
        };
        let rows = match value {
            Value::Array(rows) => rows,
            other => vec![other],
        };
        for n in rows.iter().filter_map(|r| r["number"].as_u64()) {
            if self.seen.insert(n) {
                self.numbers.push(n);
            }
        }
    }
}

/// The `decide` tool's arguments.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DecideArgs {
    /// The question about the issue, e.g. "Does the reporter run an
    /// outdated Kubernetes version?".
    pub question: String,
    /// `yes_no`, or `choice` among `options`.
    pub kind: DecideKind,
    /// For `choice`: option key to a short description of the option, e.g.
    /// {"sig/node": "kubelet, pods", "sig/network": "services, DNS"}.
    #[serde(default)]
    pub options: Option<BTreeMap<String, String>>,
}

#[derive(Debug, Clone, Copy, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum DecideKind {
    YesNo,
    Choice,
}

pub fn decide_definition() -> ToolDefinition {
    ToolDefinition::from_schema::<DecideArgs>(DECIDE).description(
        "Ask a calibrated decision model a yes/no or multiple-choice question about this issue. \
         Returns the answer with its probability. Use it when you are unsure about a judgement.",
    )
}

/// Runs one `decide` call. Like [`tools::call`], problems with the
/// arguments or the question become an error result for the model; only a
/// retryable model failure is an `Err`, so the journal retries it. The
/// usage counts a decision call, not a tool call.
pub async fn decide(tool: &DecisionTool, arguments: &str) -> Result<ToolOutput, CallError> {
    let args: DecideArgs = match serde_json::from_str(arguments) {
        Ok(args) => args,
        Err(e) => return Ok(error_output(format!("invalid arguments: {e}"))),
    };
    let request = match decide_request(tool, args) {
        Ok(request) => request,
        Err(message) => return Ok(error_output(message)),
    };
    match models::decide(DECIDE, request).await {
        Ok(summary) => {
            let content = match summary.answers.get("answer") {
                Some(DecisionAnswer::YesNo { probability }) => json!({
                    "answer": if *probability >= 0.5 { "yes" } else { "no" },
                    "probability_yes": round(*probability),
                }),
                Some(
                    a @ DecisionAnswer::Choice {
                        chosen,
                        probabilities,
                        ..
                    },
                ) => json!({
                    "answer": chosen,
                    "probability": round(a.top_probability()),
                    "probabilities": probabilities
                        .iter()
                        .map(|(k, p)| (k.clone(), round(*p)))
                        .collect::<BTreeMap<_, _>>(),
                }),
                _ => json!({"answer": null}),
            };
            Ok(ToolOutput {
                content: content.to_string(),
                is_error: false,
                usage: summary.usage,
            })
        }
        Err(e) if e.is_terminal() => Ok(error_output(format!("the decision model failed: {e}"))),
        Err(e) => Err(e),
    }
}

fn decide_request(tool: &DecisionTool, args: DecideArgs) -> Result<DecisionRequest, String> {
    let question = args.question.trim();
    if question.is_empty() {
        return Err("the question is empty".into());
    }
    let question: Question = match args.kind {
        DecideKind::YesNo => Question::yes_no(question).into(),
        DecideKind::Choice => {
            let options = args.options.unwrap_or_default();
            let options: Vec<(String, String)> = options
                .into_iter()
                .map(|(k, d)| (k.trim().to_owned(), d.trim().to_owned()))
                .filter(|(k, _)| !k.is_empty())
                .collect();
            if options.len() < 2 {
                return Err("a choice needs at least two options".into());
            }
            if options.len() > super::decision_triage::MAX_CHOICE_OPTIONS {
                return Err(format!(
                    "a choice may have at most {} options",
                    super::decision_triage::MAX_CHOICE_OPTIONS
                ));
            }
            let mut choice = Question::choice(question);
            for (key, description) in options {
                // System One needs a description for every option.
                let description = if description.is_empty() {
                    key.clone()
                } else {
                    description
                };
                choice = choice.option(key, description);
            }
            choice.into()
        }
    };
    Ok(DecisionRequest::new(&tool.model, tool.input.clone()).question("answer", question))
}

fn round(p: f64) -> f64 {
    (p * 1000.0).round() / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool() -> DecisionTool {
        DecisionTool {
            model: "ollama:tev1:0.8b-q8_0".into(),
            input: DecisionInput::text("kubelet panics"),
        }
    }

    #[test]
    fn decide_requests_are_checked() {
        let args = |v: Value| serde_json::from_value::<DecideArgs>(v).unwrap();
        let r = decide_request(
            &tool(),
            args(json!({"question": "Is it a regression?", "kind": "yes_no"})),
        )
        .unwrap();
        assert!(matches!(
            r.find_question("answer"),
            Some(Question::YesNo(_))
        ));

        let r = decide_request(
            &tool(),
            args(json!({"question": "Which SIG?", "kind": "choice",
                        "options": {"sig/node": "kubelet", "sig/network": ""}})),
        )
        .unwrap();
        r.validate(&allama::DecisionLimits::new()).unwrap();

        let err = decide_request(
            &tool(),
            args(json!({"question": "Which?", "kind": "choice", "options": {"a": "x"}})),
        )
        .unwrap_err();
        assert!(err.contains("two options"), "{err}");
        assert!(decide_request(&tool(), args(json!({"question": " ", "kind": "yes_no"}))).is_err());
    }

    #[tokio::test]
    async fn malformed_decide_arguments_are_a_message() {
        for arguments in ["{not json", r#"{"question": "x", "kind": "maybe"}"#] {
            let out = decide(&tool(), arguments).await.unwrap();
            assert!(out.is_error, "{arguments}");
            assert!(
                out.content.starts_with("error: invalid arguments"),
                "{}",
                out.content
            );
            assert_eq!(out.usage, Usage::default());
        }
    }

    #[test]
    fn decide_definition_has_an_object_schema() {
        let d = decide_definition();
        assert_eq!(d.parameters["type"], "object");
        assert!(d.parameters["properties"]["kind"].is_object());
    }

    #[test]
    fn search_results_render_as_lines() {
        let content = json!([
            {"number": 100, "title": "kubelet crashes", "created_at": "2025-01-01T00:00:00Z",
             "snippet": "panic:\n nil   pointer"},
        ])
        .to_string();
        assert_eq!(
            render(tools::SEARCH_ISSUES, &content),
            "#100 kubelet crashes (2025-01-01): panic: nil pointer\n"
        );
        assert_eq!(render(tools::SEARCH_ISSUES, "[]"), "no matching issues");
        let issue =
            json!({"number": 7, "title": "t", "body": "b", "created_at": "2025"}).to_string();
        assert_eq!(render(tools::GET_ISSUE, &issue), "#7 t (2025)\nb");
        assert_eq!(render(tools::GET_ISSUE, "error: nope"), "error: nope");
    }

    #[test]
    fn found_issues_are_deduplicated_in_order() {
        let mut found = FoundIssues::default();
        found.collect(
            tools::SEARCH_ISSUES,
            &json!([{"number": 5}, {"number": 3}]).to_string(),
        );
        found.collect(tools::GET_ISSUE, &json!({"number": 3}).to_string());
        found.collect(tools::GET_ISSUE, &json!({"number": 9}).to_string());
        found.collect(tools::LIST_LABELS, &json!([{"number": 1}]).to_string());
        assert_eq!(found.numbers, vec![5, 3, 9]);
    }

    #[test]
    fn final_prompt_lists_findings() {
        let finish = Finish::new("sys", "## Issue\n\nx\n", "Answer now.");
        let p = final_prompt(
            &finish,
            &[
                Finding::Note("looked for duplicates".into()),
                Finding::Tool {
                    name: "search_issues".into(),
                    arguments: r#"{"query":"kubelet"}"#.into(),
                    result: "#100 kubelet crashes".into(),
                },
            ],
        );
        assert!(p.starts_with("## Issue\n\nx\n\n## Findings"), "{p}");
        assert!(p.contains("Note: looked for duplicates"));
        assert!(p.contains("Called search_issues {\"query\":\"kubelet\"}:\n#100 kubelet crashes"));
        assert!(p.ends_with("Answer now."));
        assert!(final_prompt(&finish, &[]).contains("no tools were used"));
    }
}
