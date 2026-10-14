//! `Agent`: a tool-using agent loop (see [`super::agent_loop`]), each turn
//! and each tool call in its own `ctx.run`.
//!
//! Config: `model`, `decision_tool?` (gives the agent a `decide` tool backed
//! by a decision model), `max_turns`, `tools?`. Tools read the local corpus
//! snapshot, limited to issues created before `repo.as_of`: `search_issues`,
//! `get_issue`, `list_labels`, `get_issue_templates`, `get_releases`. Expected
//! to win on duplicates and lose on cost.
//!
//! `tools` picks a subset of them (absent: all). With `tools: []` and no
//! decision tool there is nothing to investigate with, so the agent is just
//! the final structured call: SingleShot's prompt answered through the
//! agent's answer type, the "agent without tools" baseline.
//!
//! The agent investigates (mostly: searches for earlier reports of the same
//! problem), then one structured-output call fills the triage. That call sees
//! SingleShot's prompt plus the agent's findings, so the difference between
//! the two strategies is the investigation, not the wording. Duplicates are
//! only kept if a tool actually returned them, which rules out issue numbers
//! the model made up.

use std::collections::BTreeSet;

use restate_sdk::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tracing::{Instrument, info};
use triage_core::config::{AgentConfig, SingleShotConfig};
use triage_core::{Field, Issue, Repo, Triage, TriageRequest, TriageResponse};

use super::agent_loop::{self, AgentOutcome, AgentSpec, DecisionTool, Finish};
use super::decision_triage::{self, truncate};
use super::single_shot::{self, ModelTriage};
use super::{parse_config, record_run, response};
use crate::models::{self, ToolDefinition};
use crate::tools::{self, ToolContext};
use crate::trace::handler_span;

pub const VERSION: &str = "4";
const NAME: &str = "Agent";
const STEP: &str = "agent";

/// The issue body in the investigation prompt. Shorter than SingleShot's,
/// because tool results need room in the same context window.
const MAX_BODY_CHARS: usize = 3000;
/// The most duplicate candidates kept.
const MAX_DUPLICATES: usize = 5;

pub struct AgentStrategy;

/// Timeouts: see "Restate timeouts" in the [`super`] module docs.
#[restate_sdk::workflow(name = "Agent", inactivity_timeout = "5m", abort_timeout = "1h")]
impl AgentStrategy {
    #[handler]
    async fn run(
        &self,
        ctx: WorkflowContext<'_>,
        Json(request): Json<TriageRequest>,
    ) -> Result<Json<TriageResponse>, HandlerError> {
        let span = handler_span(NAME, ctx.key(), ctx.headers());
        async move {
            let (request, config_hash) = parse_config::<AgentConfig>(request)?;
            record_run(&tracing::Span::current(), VERSION, &config_hash);
            let config = &request.config;
            check_address(&config.model)?;
            if let Some(model) = &config.decision_tool {
                check_address(model)?;
            }

            let corpus_tools = corpus_tools(config)?;
            let tool_ctx = ToolContext::new(request.repo.clone(), request.issue.number);
            let spec = spec(config, corpus_tools, &request.issue, &request.repo);
            let finish = finish(&request.issue, &request.repo, &config.model, &spec);
            let out: AgentOutcome<AgentAnswer> =
                agent_loop::run(&ctx, &tool_ctx, &spec, &finish).await?;
            let triage = to_triage(&out, &request.issue, &request.repo);
            info!(
                kind = triage.kind.as_ref().map(|k| k.value.as_str()),
                area = triage.area.as_ref().map(|a| a.value.as_str()),
                duplicates = ?triage.duplicates.as_ref().map(|d| &d.value),
                turns = out.turns,
                cost_usd = out.usage.cost_usd,
                "agent triage done"
            );
            Ok(Json(response(
                NAME,
                VERSION,
                config_hash,
                triage,
                out.usage,
            )))
        }
        .instrument(span)
        .await
    }
}

/// A bad model address fails the same way on every retry: a 400, before
/// anything is called.
pub(crate) fn check_address(address: &str) -> Result<(), TerminalError> {
    models::parse_address(address)
        .map(|_| ())
        .map_err(|e| TerminalError::new_with_code(400, format!("invalid config: {e}")))
}

/// The corpus tools the config allows, in the config's order. An unknown
/// name fails the same way on every retry: a 400.
pub(crate) fn corpus_tools(config: &AgentConfig) -> Result<Vec<ToolDefinition>, TerminalError> {
    let all = tools::definitions();
    let Some(names) = &config.tools else {
        return Ok(all);
    };
    names
        .iter()
        .map(|name| {
            all.iter()
                .find(|d| d.name == *name)
                .cloned()
                .ok_or_else(|| {
                    let known: Vec<&str> = all.iter().map(|d| d.name.as_str()).collect();
                    TerminalError::new_with_code(
                        400,
                        format!(
                            "invalid config: unknown tool `{name}`; available: {}",
                            known.join(", ")
                        ),
                    )
                })
        })
        .collect()
}

/// SingleShot's answer plus the duplicates the agent found.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AgentAnswer {
    #[serde(flatten)]
    pub triage: ModelTriage,
    /// Earlier issues (numbers from the findings) that report the same
    /// problem, most likely first; empty if none does.
    pub duplicates: Vec<u64>,
}

/// The investigation's system prompt, describing only the offered tools.
fn system_prompt(repo: &Repo, max_turns: u32, tools: &[ToolDefinition], decide: bool) -> String {
    let offered = |name: &str| tools.iter().any(|d| d.name == name);
    let mut p = format!(
        "You are an experienced maintainer of {} triaging a newly filed GitHub issue. \
         Investigate before you conclude, using the tools:\n",
        repo.name
    );
    if offered(tools::SEARCH_ISSUES) {
        p.push_str(
            "- search_issues: look for earlier issues that report the same problem (possible \
             duplicates). Search for the key error message or component names, not the whole \
             title.\n",
        );
    }
    if offered(tools::GET_ISSUE) {
        p.push_str(
            "- get_issue: read a promising earlier issue to check it is really the same problem.\n",
        );
    }
    let reference: Vec<&str> = [
        tools::LIST_LABELS,
        tools::GET_ISSUE_TEMPLATES,
        tools::GET_RELEASES,
    ]
    .into_iter()
    .filter(|n| offered(n))
    .collect();
    if !reference.is_empty() {
        p.push_str(&format!(
            "- {}: check the SIG labels, the issue templates, or whether the reported version is \
             current.\n",
            reference.join(", ")
        ));
    }
    if decide {
        p.push_str(
            "- decide: ask a calibrated decision model a yes/no or multiple-choice question \
             about the issue when you are unsure about a judgement.\n",
        );
    }
    p.push_str(&format!(
        "Make a few targeted tool calls; you have at most {max_turns} turns. When you are done, \
         reply without calling a tool: a few sentences on what kind of issue this is, which SIG \
         owns it, and which earlier issues (by number) report the same problem, if any."
    ));
    p
}

/// The issue as the agent sees it while investigating.
pub(crate) fn issue_prompt(issue: &Issue, repo: &Repo, max_body_chars: usize) -> String {
    format!(
        "Repository: {}\nLatest release when the issue was filed: {}\n\n\
         ## Issue #{}\n\nTitle: {}\nAuthor association: {}\nFiled: {}\n\n{}\n",
        repo.name,
        repo.latest_release.as_deref().unwrap_or("unknown"),
        issue.number,
        issue.title,
        issue.author_association,
        issue.created_at,
        truncate(&issue.body, max_body_chars),
    )
}

fn spec(
    config: &AgentConfig,
    corpus_tools: Vec<ToolDefinition>,
    issue: &Issue,
    repo: &Repo,
) -> AgentSpec {
    let decision_tool = config.decision_tool.as_ref().map(|model| DecisionTool {
        model: model.clone(),
        input: decision_triage::input(issue, repo),
    });
    let task = if corpus_tools.iter().any(|d| d.name == tools::SEARCH_ISSUES) {
        "Investigate this issue, starting with a search for earlier reports of the same problem."
    } else {
        "Investigate this issue."
    };
    AgentSpec::new(
        STEP,
        &config.model,
        system_prompt(
            repo,
            config.max_turns,
            &corpus_tools,
            decision_tool.is_some(),
        ),
        format!("{}\n{task}", issue_prompt(issue, repo, MAX_BODY_CHARS)),
    )
    .tools(corpus_tools)
    .decision_tool(decision_tool)
    .max_turns(config.max_turns)
}

const FINAL_SYSTEM_PROMPT: &str = "You are an experienced maintainer triaging newly filed \
GitHub issues. Fill in every field of the JSON answer from the issue and the findings of your \
investigation, following the field guide. Judge only from what they say; do not invent facts. \
Confidence values are numbers from 0 to 1 saying how likely your answer is to be right.";

const FINAL_INSTRUCTIONS: &str = "duplicates: the numbers of earlier issues from the findings \
that report the same problem as this issue, most likely first. Use only numbers that appear in \
the findings, and [] if none of them is the same problem (merely related issues are not \
duplicates).";

/// Without tools there is no investigation: the final call answers from the
/// issue alone, and has no earlier issues to name.
const NO_TOOLS_SYSTEM_PROMPT: &str = "You are an experienced maintainer triaging newly filed \
GitHub issues. Fill in every field of the JSON answer from the issue, following the field guide. \
Judge only from what the issue says; do not invent facts. Confidence values are numbers from 0 \
to 1 saying how likely your answer is to be right.";

const NO_TOOLS_INSTRUCTIONS: &str = "duplicates: [] (no search for earlier issues was made).";

fn finish(issue: &Issue, repo: &Repo, model: &str, spec: &AgentSpec) -> Finish {
    let (system, instructions) = if spec.offered_tools().is_empty() {
        (NO_TOOLS_SYSTEM_PROMPT, NO_TOOLS_INSTRUCTIONS)
    } else {
        (FINAL_SYSTEM_PROMPT, FINAL_INSTRUCTIONS)
    };
    Finish::new(system, single_shot_prompt(issue, repo, model), instructions)
}

/// SingleShot's user prompt: the issue and the field guide.
fn single_shot_prompt(issue: &Issue, repo: &Repo, model: &str) -> String {
    let request = TriageRequest {
        config: SingleShotConfig {
            model: model.to_owned(),
            temperature: None,
        },
        issue: issue.clone(),
        repo: repo.clone(),
    };
    single_shot::chat_request(&request)
        .messages
        .iter()
        .find_map(|m| m.text())
        .unwrap_or_default()
}

/// SingleShot's mapping, plus the duplicates: only issue numbers a tool
/// returned, never the issue itself. `duplicates` is left out when the agent
/// never searched, since then it didn't attempt them.
fn to_triage(out: &AgentOutcome<AgentAnswer>, issue: &Issue, repo: &Repo) -> Triage {
    let mut triage = single_shot::to_triage(out.answer.triage.clone(), repo);
    triage.duplicates = out.searched().then(|| {
        Field::new(valid_duplicates(
            &out.answer.duplicates,
            &out.found_issues,
            issue.number,
        ))
        .with_evidence(format!(
            "{} earlier issues seen in {} tool calls",
            out.found_issues.len(),
            out.tool_calls.len()
        ))
    });
    triage
}

pub(crate) fn valid_duplicates(answer: &[u64], found: &[u64], own: u64) -> Vec<u64> {
    let mut seen = BTreeSet::new();
    answer
        .iter()
        .copied()
        .filter(|n| *n != own && found.contains(n) && seen.insert(*n))
        .take(MAX_DUPLICATES)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategies::agent_loop::ToolRecord;
    use triage_core::{Kind, Label, Usage};

    fn repo() -> Repo {
        Repo {
            name: "kubernetes/kubernetes".into(),
            labels: vec![Label {
                name: "sig/node".into(),
                description: None,
            }],
            latest_release: Some("v1.33.2".into()),
            as_of: "2025-07-01T00:00:00Z".into(),
        }
    }

    fn issue() -> Issue {
        Issue {
            number: 42,
            title: "kubelet panics".into(),
            body: "x".repeat(MAX_BODY_CHARS + 5),
            author_association: "NONE".into(),
            created_at: "2025-07-01T00:00:00Z".into(),
        }
    }

    fn answer(duplicates: Vec<u64>) -> AgentAnswer {
        let triage: ModelTriage = serde_json::from_value(serde_json::json!({
            "summary": "kubelet panics", "kind": "bug", "kind_confidence": 0.9,
            "area": "sig/node", "area_confidence": 0.8,
            "severity": "important_soon", "severity_confidence": 0.5,
            "is_regression": false, "security_sensitive": false, "quality": 3,
            "kubernetes_version": "", "has_version": false, "has_environment": false,
            "has_steps": true, "has_expected_actual": true, "has_minimal_repro": false,
            "has_logs": true, "enough_info": true, "scope": "single", "confidence": 0.7,
            "suggested_reply": "Thanks!"
        }))
        .unwrap();
        AgentAnswer { triage, duplicates }
    }

    fn outcome(duplicates: Vec<u64>, searched: bool) -> AgentOutcome<AgentAnswer> {
        AgentOutcome {
            answer: answer(duplicates),
            usage: Usage::default(),
            turns: 2,
            hit_max_turns: false,
            tool_calls: vec![ToolRecord {
                name: if searched {
                    tools::SEARCH_ISSUES
                } else {
                    tools::LIST_LABELS
                }
                .into(),
                arguments: "{}".into(),
                is_error: false,
            }],
            found_issues: if searched { vec![10, 20, 30] } else { vec![] },
        }
    }

    #[test]
    fn duplicates_must_come_from_the_tools() {
        assert_eq!(
            valid_duplicates(&[20, 99, 42, 20, 10], &[10, 20, 42], 42),
            vec![20, 10]
        );
        let t = to_triage(&outcome(vec![30, 77], true), &issue(), &repo());
        assert_eq!(t.duplicates.unwrap().value, vec![30]);
        assert_eq!(t.kind.unwrap().value, Kind::Bug);
        assert_eq!(t.area.unwrap().value, "sig/node");
        assert!(t.suggested_reply.is_some());
        // No search: duplicates were not attempted.
        let t = to_triage(&outcome(vec![30], false), &issue(), &repo());
        assert!(t.duplicates.is_none());
    }

    #[test]
    fn answer_schema_is_flat() {
        let schema = schemars::schema_for!(AgentAnswer);
        let v = schema.as_value();
        assert_eq!(v["properties"]["kind"]["enum"][0], "bug", "{v}");
        assert_eq!(v["properties"]["duplicates"]["type"], "array");
        let required = v["required"].as_array().unwrap();
        assert!(required.iter().any(|r| r == "duplicates"));
        assert!(required.iter().any(|r| r == "summary"));
    }

    fn config(tools: Option<&[&str]>, decision_tool: bool) -> AgentConfig {
        AgentConfig {
            model: "ollama:qwen3:1.7b".into(),
            decision_tool: decision_tool.then(|| "ollama:tev1:0.8b-q8_0".into()),
            max_turns: 4,
            tools: tools.map(|t| t.iter().map(|s| s.to_string()).collect()),
        }
    }

    #[test]
    fn tools_switch() {
        let names = |c: &AgentConfig| -> Vec<String> {
            corpus_tools(c)
                .unwrap()
                .into_iter()
                .map(|d| d.name)
                .collect()
        };
        assert_eq!(names(&config(None, false)).len(), 5);
        assert!(names(&config(Some(&[]), false)).is_empty());
        assert_eq!(
            names(&config(Some(&["get_issue", "search_issues"]), false)),
            ["get_issue", "search_issues"]
        );
        let err = corpus_tools(&config(Some(&["search_issue"]), false)).unwrap_err();
        assert_eq!(err.code(), 400);

        // No tools: nothing is offered, so the loop is skipped, and the final
        // call is told there are no findings to take duplicates from.
        let c = config(Some(&[]), false);
        let sp = spec(&c, corpus_tools(&c).unwrap(), &issue(), &repo());
        assert!(sp.offered_tools().is_empty());
        let finish = finish(&issue(), &repo(), &c.model, &sp);
        assert_eq!(finish.system, NO_TOOLS_SYSTEM_PROMPT);
        assert!(finish.instructions.contains("no search"));
        // A decision tool alone still makes a loop.
        let c = config(Some(&[]), true);
        let sp = spec(&c, corpus_tools(&c).unwrap(), &issue(), &repo());
        assert_eq!(sp.offered_tools().len(), 1);

        // A subset: the prompts mention only what is offered.
        let c = config(Some(&["get_releases"]), false);
        let sp = spec(&c, corpus_tools(&c).unwrap(), &issue(), &repo());
        assert!(!sp.system.contains("search_issues"), "{}", sp.system);
        assert!(sp.system.contains("- get_releases: check"));
        assert!(sp.prompt.ends_with("Investigate this issue."));
    }

    #[test]
    fn config_tools_parse() {
        let c: AgentConfig = serde_json::from_str(r#"{"model":"ollama:qwen3:1.7b"}"#).unwrap();
        assert!(c.tools.is_none());
        let c: AgentConfig =
            serde_json::from_str(r#"{"model":"ollama:qwen3:1.7b","tools":[]}"#).unwrap();
        assert_eq!(c.tools, Some(vec![]));
    }

    #[test]
    fn prompts() {
        let config = config(None, true);
        let spec = spec(&config, corpus_tools(&config).unwrap(), &issue(), &repo());
        assert!(spec.system.contains("at most 4 turns"));
        assert!(spec.system.contains("- decide:"));
        assert!(spec.prompt.contains("## Issue #42"));
        assert!(spec.prompt.contains("[... truncated]"));
        let names: Vec<String> = spec.offered_tools().into_iter().map(|d| d.name).collect();
        assert_eq!(names.len(), 6);
        assert_eq!(names.last().unwrap(), agent_loop::DECIDE);

        assert!(spec.system.contains("- search_issues:"));
        assert!(
            spec.system
                .contains("- list_labels, get_issue_templates, get_releases: check")
        );
        let finish = finish(&issue(), &repo(), &config.model, &spec);
        assert_eq!(finish.system, FINAL_SYSTEM_PROMPT);
        assert!(
            finish.context.contains("## Field guide"),
            "{}",
            finish.context
        );
        assert!(finish.context.contains("- sig/node"));
    }
}
