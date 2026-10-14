//! `Workflow`: one decision-model call answering every typed field over the
//! issue JSON, then per-field routing in plain Rust on confidence, then
//! targeted LLM calls or a sub-agent only for the low-confidence fields, then
//! an optional reply.
//!
//! Config: `decision_model` (Jev/Clef/Tev1), `escalation: none | {model,
//! thresholds, sub_agent}`, `reply_model?`. With `escalation: none` it is the
//! decision-model-only setup. The key comparison is `Workflow` against
//! `DraftVerify`: escalation coordinated by the workflow versus escalation
//! coordinated by an agent.
//!
//! Steps:
//! 1. One decision call drafts every typed field ([`decision_triage`]).
//! 2. With escalation, [`plan`] picks the fields below their threshold.
//!    Each gets its own small structured LLM call (`escalate:<field>`) that
//!    sees only the issue, that field's guide and the decision model's
//!    probabilities for it as a hint. The calls run one after another,
//!    each in its own `ctx.run`: restate-sdk 0.12 can't replay concurrent
//!    runs (see below).
//! 3. With `sub_agent`, a tool-using agent loop (`sub_agent`) searches the
//!    corpus for duplicates, which a decision model can't produce, and takes
//!    `area` over from the targeted calls when it is unsure, since similar
//!    earlier issues show which component is involved.
//! 4. With `reply_model`, one call writes `suggested_reply` from the final
//!    triage.
//!
//! Every escalated field's evidence starts with `escalated: <model>`; its
//! confidence is the escalation model's own, which (unlike the decision
//! model's) is self-reported. A failed escalation call keeps the decision
//! model's answer rather than failing the triage.
//!
//! Why not concurrent: several `ctx.run` futures polled together do start
//! together, but the SDK flushes a run's result to Restate only when another
//! run starts, so finished results sit in the buffer; once Restate's
//! inactivity timeout closes the stream, the invocation suspends, and on
//! resume the replay fails (`'await' could not be replayed`, code 570) as
//! soon as a run that isn't the last one journaled is still open. Seen live
//! with three concurrent voters; it retried forever.

use std::fmt::Write;

use restate_sdk::prelude::*;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tracing::{Instrument, info, warn};
use triage_core::config::{Escalation, WorkflowConfig};
use triage_core::{
    Field, Issue, Kind, QUALITY_RUBRIC, Repo, ReproItem, Reproducibility, Scope, Severity, Triage,
    TriageRequest, TriageResponse, Usage,
};

use super::agent::{check_address, issue_prompt, valid_duplicates};
use super::agent_loop::{self, AgentOutcome, AgentSpec, Finish};
use super::decision_triage::{self, bug_like, escalatable_fields, match_sig, q, truncate};
use super::heuristic::K8sVersion;
use super::{parse_config, record_run, response};
use crate::models::{
    self, ChatRequest, ChatTurn, DecisionAnswer, DecisionSummary, Structured, journal,
};
use crate::tools::{self, ToolContext};
use crate::trace::handler_span;

pub const VERSION: &str = "4";
const NAME: &str = "Workflow";
const SUB_AGENT_STEP: &str = "sub_agent";
const REPLY_STEP: &str = "reply";

/// The issue body in the escalation and reply prompts. They ask one thing
/// each, so the body gets most of a small model's context window.
const MAX_BODY_CHARS: usize = 4000;
/// The sub-agent's prompt leaves room for tool results.
const SUB_AGENT_BODY_CHARS: usize = 3000;
const SUB_AGENT_MAX_TURNS: u32 = 4;
/// Ceilings with room for a reasoning model's reasoning tokens; see agent_loop.
const ESCALATE_OUTPUT_TOKENS: u32 = 4000;
const REPLY_OUTPUT_TOKENS: u32 = 4000;
/// Options shown in a choice hint; the long tail says nothing.
const HINT_OPTIONS: usize = 5;
/// The escalation model's evidence is cut to this in the field's evidence.
const MAX_EVIDENCE_CHARS: usize = 300;

pub struct WorkflowStrategy;

/// Timeouts: see "Restate timeouts" in the [`super`] module docs.
#[restate_sdk::workflow(name = "Workflow", inactivity_timeout = "5m", abort_timeout = "1h")]
impl WorkflowStrategy {
    #[handler]
    async fn run(
        &self,
        ctx: WorkflowContext<'_>,
        Json(request): Json<TriageRequest>,
    ) -> Result<Json<TriageResponse>, HandlerError> {
        let span = handler_span(NAME, ctx.key(), ctx.headers());
        async move {
            let (request, config_hash) = parse_config::<WorkflowConfig>(request)?;
            record_run(&tracing::Span::current(), VERSION, &config_hash);
            let config = &request.config;
            check_address(&config.decision_model)?;
            if let Some(escalation) = &config.escalation {
                check_address(&escalation.model)?;
            }
            if let Some(model) = &config.reply_model {
                check_address(model)?;
            }
            let (issue, repo) = (&request.issue, &request.repo);

            let (draft, summary) =
                decision_triage::run(&ctx, &config.decision_model, issue, repo).await?;
            let mut usage = summary.usage.clone();
            let mut triage = draft.clone();

            if let Some(escalation) = &config.escalation {
                let plan = plan(&draft, escalation, repo);
                info!(
                    escalate.fields = plan.targeted.join(","),
                    escalate.sub_agent = plan.sub_agent.is_some(),
                    "routing done"
                );
                for &field in &plan.targeted {
                    let result = targeted(
                        &ctx,
                        field,
                        &escalation.model,
                        issue,
                        repo,
                        &summary,
                        &draft,
                    )
                    .await;
                    match result {
                        Ok(escalated) => {
                            usage += &escalated.usage;
                            escalated.apply(&mut triage);
                        }
                        // The failure is journaled, so every replay takes
                        // this branch too.
                        Err(e) => warn!(
                            field,
                            error = %e,
                            "escalation failed; keeping the decision model's answer"
                        ),
                    }
                }
                if let Some(task) = plan.sub_agent {
                    let tool_ctx = ToolContext::new(repo.clone(), issue.number);
                    let (spec, finish) =
                        sub_agent_spec(&escalation.model, task, issue, repo, &summary);
                    // Counted whether or not the loop succeeds: a failed
                    // final call still leaves its earlier turns paid for.
                    let mut spent = Usage::default();
                    let result = agent_loop::run_metered::<SubAgentAnswer>(
                        &ctx, &tool_ctx, &spec, &finish, &mut spent,
                    )
                    .await;
                    usage += &spent;
                    match result {
                        Ok(out) => {
                            apply_sub_agent(
                                &mut triage,
                                &out,
                                task,
                                &escalation.model,
                                issue,
                                repo,
                            );
                        }
                        Err(e) => warn!(error = %e, "sub-agent failed; keeping the draft"),
                    }
                }
                drop_checklist_unless_bug_like(&mut triage);
            }

            if let Some(model) = &config.reply_model {
                match reply(&ctx, model, issue, repo, &triage).await {
                    Ok(Some((field, reply_usage))) => {
                        usage += &reply_usage;
                        triage.suggested_reply = Some(field);
                    }
                    Ok(None) => {}
                    Err(e) => warn!(error = %e, "reply failed; leaving it out"),
                }
            }

            info!(
                kind = triage.kind.as_ref().map(|k| k.value.as_str()),
                area = triage.area.as_ref().map(|a| a.value.as_str()),
                cost_usd = usage.cost_usd,
                "workflow triage done"
            );
            Ok(Json(response(NAME, VERSION, config_hash, triage, usage)))
        }
        .instrument(span)
        .await
    }
}

/// Which fields go where.
#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    /// One targeted LLM call each, in `decision_triage::FIELDS` order.
    pub targeted: Vec<&'static str>,
    pub sub_agent: Option<SubAgentTask>,
}

/// What the sub-agent is asked for. It always looks for duplicates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SubAgentTask {
    pub area: bool,
}

/// Routes the draft's fields: those below their threshold are escalated,
/// except where escalating can't change the outcome (see
/// [`decision_triage::escalatable_fields`]). With `sub_agent`, `area` goes
/// to the sub-agent, which also always looks for duplicates.
pub fn plan(draft: &Triage, escalation: &Escalation, repo: &Repo) -> Plan {
    let mut targeted = escalatable_fields(draft, &escalation.thresholds, repo);
    let sub_agent = escalation.sub_agent.then(|| {
        let area = targeted.contains(&"area");
        targeted.retain(|&f| f != "area");
        SubAgentTask { area }
    });
    Plan {
        targeted,
        sub_agent,
    }
}

/// Like the other strategies: no checklist for kinds that aren't bug-like,
/// which escalating `kind` may have turned the issue into.
fn drop_checklist_unless_bug_like(triage: &mut Triage) {
    if !bug_like(triage) {
        triage.reproducibility = None;
    }
}

/// What one targeted call decided about its field.
#[derive(Debug, Clone)]
struct Escalated {
    field: &'static str,
    /// Only `field` is set, or nothing when the answer wasn't valid.
    overrides: Triage,
    /// The escalation found the field can't be answered (no version stated
    /// for `outdated_version`): the draft's guess is removed, not kept.
    clear: bool,
    usage: Usage,
}

impl Escalated {
    fn apply(self, triage: &mut Triage) {
        if self.clear {
            clear_field(triage, self.field);
        } else {
            *triage = decision_triage::merge(std::mem::take(triage), self.overrides);
        }
    }
}

fn clear_field(triage: &mut Triage, name: &str) {
    match name {
        "kind" => triage.kind = None,
        "area" => triage.area = None,
        "severity" => triage.severity = None,
        "is_regression" => triage.is_regression = None,
        "security_sensitive" => triage.security_sensitive = None,
        "quality" => triage.quality = None,
        "reproducibility" => triage.reproducibility = None,
        "outdated_version" => triage.outdated_version = None,
        "scope" => triage.scope = None,
        "duplicates" => triage.duplicates = None,
        "suggested_reply" => triage.suggested_reply = None,
        _ => {}
    }
}

/// One targeted call for `field`, journaled as `escalate:<field>`. Every
/// field has its own small answer type, so the model's output is
/// constrained to just what is asked.
async fn targeted(
    ctx: &WorkflowContext<'_>,
    field: &'static str,
    model: &str,
    issue: &Issue,
    repo: &Repo,
    summary: &DecisionSummary,
    draft: &Triage,
) -> Result<Escalated, TerminalError> {
    let step = format!("escalate:{field}");
    let request = ChatRequest::new(model)
        .system(ESCALATE_SYSTEM_PROMPT)
        .user(targeted_prompt(field, issue, repo, summary))
        .max_output_tokens(ESCALATE_OUTPUT_TOKENS);
    let note = Note {
        model: model.to_owned(),
        draft: decision_triage::describe_field(draft, field),
    };
    let draft_checklist = draft.reproducibility.as_ref().map(|f| f.value.clone());
    let (overrides, clear, usage) = match field {
        "kind" => {
            let out: Structured<KindAnswer> = ask(ctx, &step, request).await?;
            let a = out.value;
            let kind =
                Kind::parse(&normalize(&a.kind)).map(|k| note.field(k, a.confidence, &a.evidence));
            (
                Triage {
                    kind,
                    ..Triage::default()
                },
                false,
                out.usage,
            )
        }
        "area" => {
            let out: Structured<AreaAnswer> = ask(ctx, &step, request).await?;
            let a = out.value;
            let area = match_sig(&a.area, repo).map(|s| note.field(s, a.confidence, &a.evidence));
            (
                Triage {
                    area,
                    ..Triage::default()
                },
                false,
                out.usage,
            )
        }
        "severity" => {
            let out: Structured<SeverityAnswer> = ask(ctx, &step, request).await?;
            let a = out.value;
            let severity = Severity::parse(&normalize(&a.severity))
                .map(|s| note.field(s, a.confidence, &a.evidence));
            (
                Triage {
                    severity,
                    ..Triage::default()
                },
                false,
                out.usage,
            )
        }
        "is_regression" | "security_sensitive" => {
            let out: Structured<YesNoAnswer> = ask(ctx, &step, request).await?;
            let a = out.value;
            let value = Some(note.field(a.answer, a.confidence, &a.evidence));
            let overrides = if field == "is_regression" {
                Triage {
                    is_regression: value,
                    ..Triage::default()
                }
            } else {
                Triage {
                    security_sensitive: value,
                    ..Triage::default()
                }
            };
            (overrides, false, out.usage)
        }
        "quality" => {
            let out: Structured<QualityAnswer> = ask(ctx, &step, request).await?;
            let a = out.value;
            let quality = (1..=5)
                .contains(&a.quality)
                .then(|| note.field(a.quality as u8, a.confidence, &a.evidence));
            (
                Triage {
                    quality,
                    ..Triage::default()
                },
                false,
                out.usage,
            )
        }
        "reproducibility" => {
            let out: Structured<ReproAnswer> = ask(ctx, &step, request).await?;
            let a = out.value;
            let checklist = Reproducibility::from_checks(
                a.has_version,
                a.has_environment,
                a.has_steps,
                a.has_expected_actual,
                a.has_minimal_repro,
                a.has_logs,
                a.enough_info,
            );
            let mut field = note.field(checklist, a.confidence, &a.evidence);
            if let Some(d) = draft_checklist.filter(|d| *d != field.value) {
                let _ = write!(
                    field.evidence.get_or_insert_default(),
                    " (decision model's enough_info: {})",
                    d.enough_info
                );
            }
            (
                Triage {
                    reproducibility: Some(field),
                    ..Triage::default()
                },
                false,
                out.usage,
            )
        }
        "outdated_version" => {
            let out: Structured<VersionAnswer> = ask(ctx, &step, request).await?;
            let outdated = outdated(&out.value, repo, &note);
            let clear = outdated.is_none();
            (
                Triage {
                    outdated_version: outdated,
                    ..Triage::default()
                },
                clear,
                out.usage,
            )
        }
        "scope" => {
            let out: Structured<ScopeAnswer> = ask(ctx, &step, request).await?;
            let a = out.value;
            let scope = match normalize(&a.scope).as_str() {
                "single" => Some(Scope::Single),
                "multiple" => Some(Scope::Multiple),
                _ => None,
            }
            .map(|s| note.field(s, a.confidence, &a.evidence));
            (
                Triage {
                    scope,
                    ..Triage::default()
                },
                false,
                out.usage,
            )
        }
        other => {
            return Err(TerminalError::new(format!(
                "no escalation for field `{other}`"
            )));
        }
    };
    Ok(Escalated {
        field,
        overrides,
        clear,
        usage,
    })
}

async fn ask<T>(
    ctx: &WorkflowContext<'_>,
    step: &str,
    request: ChatRequest,
) -> Result<Structured<T>, TerminalError>
where
    T: JsonSchema + Serialize + DeserializeOwned + Send + 'static,
{
    journal(ctx, step, models::extract(step, request)).await
}

/// How an escalated field is labeled.
struct Note {
    model: String,
    /// The decision model's answer, as `describe_field` shows it.
    draft: String,
}

impl Note {
    fn field<T>(&self, value: T, confidence: f64, evidence: &str) -> Field<T> {
        let mut field = Field::new(value).with_evidence(self.evidence(evidence));
        field.confidence = self_reported(confidence);
        field
    }

    fn evidence(&self, evidence: &str) -> String {
        let mut text = format!("escalated: {} (decision model: {})", self.model, self.draft);
        let evidence = evidence.trim();
        if !evidence.is_empty() {
            let _ = write!(text, ": {}", truncate(evidence, MAX_EVIDENCE_CHARS));
        }
        text
    }
}

/// Clamps a self-reported confidence into 0..=1, reading values above 1 as
/// percentages, which small models sometimes give.
fn self_reported(c: f64) -> Option<f64> {
    if !c.is_finite() || c < 0.0 {
        return None;
    }
    Some(if c > 1.0 { (c / 100.0).min(1.0) } else { c })
}

fn normalize(s: &str) -> String {
    let s = s.trim().to_lowercase().replace([' ', '-'], "_");
    s.strip_prefix("kind/").map(str::to_owned).unwrap_or(s)
}

/// The outdated check in code, from the version the model read off the
/// issue. `None` when no Kubernetes version is stated.
fn outdated(answer: &VersionAnswer, repo: &Repo, note: &Note) -> Option<Field<bool>> {
    let latest_tag = repo.latest_release.as_deref()?;
    let latest = K8sVersion::parse(latest_tag)?;
    let version = K8sVersion::parse(&answer.kubernetes_version)?;
    (version.major == 1).then(|| {
        Field::new(version.older_than(&latest)).with_evidence(note.evidence(&format!(
            "reported {}, latest {latest_tag}",
            answer.kubernetes_version.trim()
        )))
    })
}

const ESCALATE_SYSTEM_PROMPT: &str = "You are an experienced maintainer triaging a newly filed \
GitHub issue. Answer one question about it in JSON. Judge only from what the issue says; do not \
invent facts. Give the evidence first: a short quote from the issue or the reason for your \
answer. Confidence is a number from 0 to 1 saying how likely your answer is to be right.";

/// The issue, the one question, and the decision model's probabilities.
fn targeted_prompt(field: &str, issue: &Issue, repo: &Repo, summary: &DecisionSummary) -> String {
    let mut p = issue_prompt(issue, repo, MAX_BODY_CHARS);
    let _ = writeln!(p, "\n## Question\n\n{}\n", field_guide(field, issue, repo));
    let _ = writeln!(
        p,
        "## Hint\n\nA fast classifier read the issue and was unsure here. Its probabilities: \
         {}. Treat them as a hint only, and decide from the issue.",
        hint(field, summary)
    );
    p
}

fn field_guide(field: &str, issue: &Issue, repo: &Repo) -> String {
    let mut g = String::new();
    match field {
        "kind" => {
            g.push_str("What kind of issue is this? kind, one of:\n");
            for k in Kind::ALL {
                let _ = writeln!(g, "- {}: {}", k.as_str(), k.description());
            }
        }
        "area" => {
            g.push_str("Which SIG (special interest group) owns the problem? area, one of:\n");
            for (sig, hint) in decision_triage::sig_options(issue, repo) {
                let _ = writeln!(g, "- {sig}: {hint}");
            }
        }
        "severity" => {
            g.push_str("How urgently must maintainers address it? severity, one of:\n");
            for s in Severity::ALL {
                let _ = writeln!(g, "- {}: {}", s.as_str(), s.description());
            }
        }
        "is_regression" => g.push_str(
            "Is this a regression? answer true if it worked in an earlier version and broke \
             after an upgrade or change.",
        ),
        "security_sensitive" => g.push_str(
            "Is it security sensitive? answer true if it reports a vulnerability or security \
             risk (CVE, privilege escalation, leaked secrets, auth bypass).",
        ),
        "quality" => {
            g.push_str("How well is the issue written? quality, 1 to 5:\n");
            for (score, text) in QUALITY_RUBRIC {
                let _ = writeln!(g, "- {score}: {text}");
            }
        }
        "reproducibility" => g.push_str(
            "Does the report have what a maintainer needs to reproduce it? Each true or false:\n\
             - has_version: states the Kubernetes or component version\n\
             - has_environment: describes the environment (cloud, OS, runtime, install method)\n\
             - has_steps: gives steps to reproduce\n\
             - has_expected_actual: says both what was expected and what happened\n\
             - has_minimal_repro: includes a minimal reproduction (manifest, commands or code)\n\
             - has_logs: includes logs, errors or stack traces\n\
             - enough_info: a maintainer could start reproducing it with what is given\n\
             In evidence, name what is missing.",
        ),
        "outdated_version" => g.push_str(
            "Which Kubernetes version does the reporter run? kubernetes_version: the version \
             from `kubectl version` output or the text, e.g. \"v1.31.2\"; \"\" if the issue \
             doesn't say.",
        ),
        "scope" => g.push_str(
            "Is this one problem or several? scope: single (one problem or request) or \
             multiple (several unrelated problems or requests bundled together).",
        ),
        _ => {}
    }
    g.trim_end().to_owned()
}

/// The decision model's probabilities for `field`, compactly:
/// `bug 0.55, feature 0.30`, `yes 0.62`.
fn hint(field: &str, summary: &DecisionSummary) -> String {
    let yes = |name: &str| {
        summary
            .yes_no(name)
            .map(|p| format!("yes {p:.2}"))
            .unwrap_or_else(|| "no answer".to_owned())
    };
    match field {
        "kind" | "area" | "severity" => match summary.answers.get(field) {
            Some(DecisionAnswer::Choice { probabilities, .. }) => {
                let mut options: Vec<(&String, &f64)> = probabilities.iter().collect();
                options.sort_by(|a, b| b.1.total_cmp(a.1));
                options
                    .into_iter()
                    .take(HINT_OPTIONS)
                    .map(|(key, p)| format!("{key} {p:.2}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            }
            _ => "no answer".to_owned(),
        },
        "quality" => match summary.answers.get(q::QUALITY) {
            Some(DecisionAnswer::Score { probabilities, .. }) => QUALITY_RUBRIC
                .iter()
                .zip(probabilities)
                .map(|((level, _), p)| format!("{level}: {p:.2}"))
                .collect::<Vec<_>>()
                .join(", "),
            _ => "no answer".to_owned(),
        },
        "scope" => summary
            .yes_no(q::MULTIPLE_PROBLEMS)
            .map(|p| format!("multiple {p:.2}, single {:.2}", 1.0 - p))
            .unwrap_or_else(|| "no answer".to_owned()),
        "outdated_version" => format!(
            "older than the latest release: {}",
            yes(q::OUTDATED_VERSION)
        ),
        "reproducibility" => [
            q::HAS_VERSION,
            q::HAS_ENVIRONMENT,
            q::HAS_STEPS,
            q::HAS_EXPECTED_ACTUAL,
            q::HAS_MINIMAL_REPRO,
            q::HAS_LOGS,
            q::ENOUGH_INFO,
        ]
        .into_iter()
        .map(|name| format!("{name} {}", yes(name)))
        .collect::<Vec<_>>()
        .join(", "),
        other => yes(other),
    }
}

/// Answer types, one per field. Field order is generation order (the schema
/// keeps declaration order through serde_json's `preserve_order`): the
/// evidence comes first so the answer can lean on it, the confidence last.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct KindAnswer {
    evidence: String,
    #[schemars(extend("enum" = ["bug", "feature", "question", "docs", "cleanup", "failing_test", "flake", "other", "spam"]))]
    kind: String,
    confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct AreaAnswer {
    evidence: String,
    /// One of the SIG labels from the question, e.g. "sig/node".
    area: String,
    confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct SeverityAnswer {
    evidence: String,
    #[schemars(extend("enum" = ["critical", "important_soon", "important_longterm", "backlog"]))]
    severity: String,
    confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct YesNoAnswer {
    evidence: String,
    answer: bool,
    confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct QualityAnswer {
    evidence: String,
    /// 1 to 5.
    quality: i64,
    confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct ReproAnswer {
    /// What is missing from the report.
    evidence: String,
    has_version: bool,
    has_environment: bool,
    has_steps: bool,
    has_expected_actual: bool,
    has_minimal_repro: bool,
    has_logs: bool,
    enough_info: bool,
    confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct VersionAnswer {
    evidence: String,
    /// e.g. "v1.31.2", or "" if the issue doesn't say.
    kubernetes_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct ScopeAnswer {
    evidence: String,
    #[schemars(extend("enum" = ["single", "multiple"]))]
    scope: String,
    confidence: f64,
}

/// The sub-agent's final answer.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct SubAgentAnswer {
    /// Earlier issues from the findings that report the same problem, most
    /// likely first.
    #[serde(default)]
    pub duplicates: Vec<u64>,
    /// One of the SIG labels, e.g. "sig/node"; null when not asked.
    #[serde(default)]
    pub area: Option<String>,
    /// 0 to 1; null when not asked.
    #[serde(default)]
    pub area_confidence: Option<f64>,
}

fn sub_agent_spec(
    model: &str,
    task: SubAgentTask,
    issue: &Issue,
    repo: &Repo,
    summary: &DecisionSummary,
) -> (AgentSpec, Finish) {
    let mut prompt = issue_prompt(issue, repo, SUB_AGENT_BODY_CHARS);
    prompt.push_str(
        "\n## Task\n\nFind earlier issues that report the same problem as this one (possible \
         duplicates): search for the key error message or component names, not the whole \
         title, and read promising results with get_issue.\n",
    );
    if task.area {
        let _ = writeln!(
            prompt,
            "\nAlso work out which SIG owns the problem; earlier reports of similar problems \
             show which component is involved. {}\nA fast classifier was unsure; its \
             probabilities: {}.",
            field_guide("area", issue, repo),
            hint("area", summary)
        );
    }
    let system = format!(
        "You are an experienced maintainer of {} helping to triage a newly filed GitHub issue. \
         Use the tools:\n\
         - search_issues: full-text search over earlier issues.\n\
         - get_issue: read one earlier issue.\n\
         Make a few targeted calls; you have at most {SUB_AGENT_MAX_TURNS} turns. When you are \
         done, reply without calling a tool: which earlier issues (by number) report the same \
         problem, if any{}.",
        repo.name,
        if task.area {
            ", and which SIG owns it"
        } else {
            ""
        }
    );
    let spec = AgentSpec::new(SUB_AGENT_STEP, model, system, prompt.clone())
        .tools(tools::definitions_for(&[
            tools::SEARCH_ISSUES,
            tools::GET_ISSUE,
        ]))
        .max_turns(SUB_AGENT_MAX_TURNS);
    let mut instructions = String::from(
        "duplicates: the numbers of earlier issues from the findings that report the same \
         problem as this issue, most likely first. Use only numbers that appear in the \
         findings, and [] if none is the same problem (merely related issues are not \
         duplicates).\n",
    );
    instructions.push_str(if task.area {
        "area: the SIG label that owns the problem, from the list above; area_confidence: 0 to \
         1, how likely that is right."
    } else {
        "area and area_confidence: null."
    });
    let finish = Finish::new(
        "You are an experienced maintainer triaging a GitHub issue. Answer in JSON from the \
         issue and the findings of the investigation. Do not invent issue numbers.",
        prompt,
        instructions,
    );
    (spec, finish)
}

fn apply_sub_agent(
    triage: &mut Triage,
    out: &AgentOutcome<SubAgentAnswer>,
    task: SubAgentTask,
    model: &str,
    issue: &Issue,
    repo: &Repo,
) {
    if task.area {
        let note = Note {
            model: model.to_owned(),
            draft: decision_triage::describe_field(triage, "area"),
        };
        if let Some(area) = out.answer.area.as_deref().and_then(|a| match_sig(a, repo)) {
            let evidence = format!("sub-agent, {} tool calls", out.tool_calls.len());
            triage.area = Some(note.field(
                area,
                out.answer.area_confidence.unwrap_or(f64::NAN),
                &evidence,
            ));
        }
    }
    if out.searched() {
        triage.duplicates = Some(
            Field::new(valid_duplicates(
                &out.answer.duplicates,
                &out.found_issues,
                issue.number,
            ))
            .with_evidence(format!(
                "escalated: {model} (sub-agent saw {} earlier issues in {} tool calls)",
                out.found_issues.len(),
                out.tool_calls.len()
            )),
        );
    }
}

const REPLY_SYSTEM_PROMPT: &str = "You are a friendly, experienced maintainer writing the first \
reply to a newly filed GitHub issue. Write plain text: 2 to 4 sentences, no headings, no \
sign-off. Do not promise fixes or timelines, and do not invent facts.";

/// Writes `suggested_reply` from the final triage, journaled as `reply`.
/// Spam gets no reply.
async fn reply(
    ctx: &WorkflowContext<'_>,
    model: &str,
    issue: &Issue,
    repo: &Repo,
    triage: &Triage,
) -> Result<Option<(Field<String>, Usage)>, TerminalError> {
    if triage.kind.as_ref().is_some_and(|k| k.value == Kind::Spam) {
        return Ok(None);
    }
    let request = ChatRequest::new(model)
        .system(REPLY_SYSTEM_PROMPT)
        .user(reply_prompt(issue, repo, triage))
        .max_output_tokens(REPLY_OUTPUT_TOKENS);
    let turn: ChatTurn = journal(ctx, REPLY_STEP, models::chat(REPLY_STEP, request)).await?;
    let text = turn
        .text
        .as_deref()
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(str::to_owned);
    Ok(text.map(|t| {
        (
            Field::new(t).with_evidence(format!("written by {model}")),
            turn.usage,
        )
    }))
}

fn reply_prompt(issue: &Issue, repo: &Repo, triage: &Triage) -> String {
    let mut p = issue_prompt(issue, repo, MAX_BODY_CHARS);
    p.push_str("\n## Triage\n\n");
    for name in ["kind", "area", "severity"] {
        let _ = writeln!(
            p,
            "- {name}: {}",
            decision_triage::describe_field(triage, name)
        );
    }
    p.push_str("\n## What to write\n\n");
    let kind = triage.kind.as_ref().map(|k| k.value);
    match kind {
        Some(Kind::Question) => p.push_str(
            "This is a support question, not a bug report or feature request. Thank the author \
             and point them to the support channels: the Kubernetes Slack (#kubernetes-users, \
             slack.k8s.io), the forum at discuss.kubernetes.io, or Stack Overflow (tag \
             kubernetes). Say that the issue tracker is for bugs and feature requests.\n",
        ),
        Some(Kind::Feature) => p.push_str(
            "This is a feature request. Thank the author for the proposal, and say that the \
             owning SIG will discuss it and that larger changes go through the KEP process \
             (kubernetes/enhancements).\n",
        ),
        Some(Kind::Docs) => p.push_str(
            "This is about documentation. Thank the author; if it concerns the kubernetes.io \
             website, suggest filing it in kubernetes/website.\n",
        ),
        Some(Kind::Bug | Kind::FailingTest | Kind::Flake) | None => {
            let missing = triage
                .reproducibility
                .as_ref()
                .map(|r| (r.value.missing.clone(), r.value.enough_info));
            match missing {
                Some((missing, enough_info)) if !missing.is_empty() || !enough_info => {
                    p.push_str(
                        "Thank the author and ask for exactly these missing details, and \
                         nothing else:\n",
                    );
                    for item in &missing {
                        let _ = writeln!(p, "- {}", repro_item(*item));
                    }
                    if missing.is_empty() {
                        p.push_str("- clearer steps to reproduce the problem\n");
                    }
                }
                _ => p.push_str(
                    "Thank the author and say the report has what is needed to start \
                     investigating, and that the owning SIG will look at it.\n",
                ),
            }
        }
        Some(_) => p.push_str("Thank the author and say that the owning SIG will triage it.\n"),
    }
    if let Some(d) = triage.duplicates.as_ref().filter(|d| !d.value.is_empty()) {
        let numbers: Vec<String> = d.value.iter().map(|n| format!("#{n}")).collect();
        let _ = writeln!(
            p,
            "Mention that {} may report the same problem.",
            numbers.join(", ")
        );
    }
    p
}

fn repro_item(item: ReproItem) -> &'static str {
    match item {
        ReproItem::Version => "the Kubernetes version (`kubectl version` output)",
        ReproItem::Environment => {
            "the environment: cloud provider or hardware, OS, container runtime, install method"
        }
        ReproItem::Steps => "the steps to reproduce the problem",
        ReproItem::ExpectedActual => "what they expected to happen and what happened instead",
        ReproItem::MinimalRepro => "a minimal reproduction (a manifest or commands)",
        ReproItem::Logs => "the relevant logs or error messages",
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use triage_core::Label;
    use triage_core::config::Thresholds;

    fn repo() -> Repo {
        Repo {
            name: "kubernetes/kubernetes".into(),
            labels: ["sig/node", "sig/network", "sig/storage"]
                .into_iter()
                .map(|name| Label {
                    name: name.into(),
                    description: None,
                })
                .collect(),
            latest_release: Some("v1.33.2".into()),
            as_of: "2025-07-01T00:00:00Z".into(),
        }
    }

    fn issue() -> Issue {
        Issue {
            number: 42,
            title: "Service DNS stops resolving".into(),
            body: "After upgrading to v1.32.1, pods can't resolve services.".into(),
            author_association: "NONE".into(),
            created_at: "2025-07-01T00:00:00Z".into(),
        }
    }

    fn summary() -> DecisionSummary {
        let mut answers = BTreeMap::new();
        answers.insert(
            q::KIND.to_owned(),
            DecisionAnswer::Choice {
                chosen: "bug".into(),
                confidence: Some(0.95),
                probabilities: [("bug".to_owned(), 0.9), ("feature".to_owned(), 0.06)].into(),
            },
        );
        answers.insert(
            q::AREA.to_owned(),
            DecisionAnswer::Choice {
                chosen: "sig/node".into(),
                confidence: None,
                probabilities: [
                    ("sig/node".to_owned(), 0.5),
                    ("sig/network".to_owned(), 0.4),
                    ("sig/storage".to_owned(), 0.1),
                ]
                .into(),
            },
        );
        answers.insert(
            q::MULTIPLE_PROBLEMS.to_owned(),
            DecisionAnswer::YesNo { probability: 0.3 },
        );
        answers.insert(
            q::HAS_LOGS.to_owned(),
            DecisionAnswer::YesNo { probability: 0.6 },
        );
        answers.insert(
            q::QUALITY.to_owned(),
            DecisionAnswer::Score {
                score: 2.0,
                most_likely: Some(2),
                confidence: None,
                probabilities: vec![0.05, 0.25, 0.4, 0.2, 0.1],
            },
        );
        DecisionSummary {
            answers,
            usage: Usage::default(),
        }
    }

    /// A draft as the decision model gives it: sure about kind and the
    /// yes/no fields, unsure about area, severity and scope.
    fn draft() -> Triage {
        Triage {
            kind: Some(Field::new(Kind::Bug).with_confidence(0.95)),
            area: Some(Field::new("sig/node".to_owned()).with_confidence(0.5)),
            severity: Some(Field::new(Severity::Backlog).with_confidence(0.4)),
            is_regression: Some(Field::new(true).with_confidence(0.9)),
            security_sensitive: Some(Field::new(false).with_confidence(0.97)),
            quality: Some(Field::new(3).with_confidence(0.85)),
            reproducibility: Some(
                Field::new(Reproducibility::from_checks(
                    true, false, false, true, false, false, true,
                ))
                .with_confidence(0.9),
            ),
            outdated_version: Some(Field::new(true).with_confidence(0.95)),
            scope: Some(Field::new(Scope::Single).with_confidence(0.7)),
            duplicates: None,
            suggested_reply: None,
        }
    }

    fn escalation(sub_agent: bool) -> Escalation {
        Escalation {
            model: "ollama:qwen3:1.7b".into(),
            thresholds: Thresholds::default(),
            sub_agent,
        }
    }

    #[test]
    fn routes_low_confidence_fields() {
        let p = plan(&draft(), &escalation(false), &repo());
        assert_eq!(p.targeted, vec!["area", "severity", "scope"]);
        assert_eq!(p.sub_agent, None);

        // With the sub-agent, area moves there; duplicates always do.
        let p = plan(&draft(), &escalation(true), &repo());
        assert_eq!(p.targeted, vec!["severity", "scope"]);
        assert_eq!(p.sub_agent, Some(SubAgentTask { area: true }));
        let mut confident = draft();
        confident.area.as_mut().unwrap().confidence = Some(0.9);
        let p = plan(&confident, &escalation(true), &repo());
        assert_eq!(p.sub_agent, Some(SubAgentTask { area: false }));

        // Per-field thresholds.
        let mut e = escalation(false);
        e.thresholds.per_field.insert("area".into(), 0.4);
        e.thresholds.per_field.insert("quality".into(), 0.9);
        assert_eq!(
            plan(&draft(), &e, &repo()).targeted,
            vec!["severity", "quality", "scope"]
        );
    }

    #[test]
    fn skips_fields_escalation_cannot_change() {
        // A confident feature request has no checklist: nothing to escalate.
        let mut feature = draft();
        feature.kind = Some(Field::new(Kind::Feature).with_confidence(0.95));
        feature.reproducibility = None;
        assert!(
            !plan(&feature, &escalation(false), &repo())
                .targeted
                .contains(&"reproducibility")
        );
        // An unsure one might turn out to be a bug, so the checklist goes too.
        feature.kind.as_mut().unwrap().confidence = Some(0.5);
        let targeted = plan(&feature, &escalation(false), &repo()).targeted;
        assert!(targeted.contains(&"kind") && targeted.contains(&"reproducibility"));

        // No latest release: nothing to compare a version against.
        let mut no_release = repo();
        no_release.latest_release = None;
        let mut d = draft();
        d.outdated_version = None;
        assert!(
            !plan(&d, &escalation(false), &no_release)
                .targeted
                .contains(&"outdated_version")
        );
        assert!(
            plan(&d, &escalation(false), &repo())
                .targeted
                .contains(&"outdated_version")
        );
    }

    #[test]
    fn hints_show_the_decision_probabilities() {
        let s = summary();
        assert_eq!(
            hint("area", &s),
            "sig/node 0.50, sig/network 0.40, sig/storage 0.10"
        );
        assert_eq!(hint("scope", &s), "multiple 0.30, single 0.70");
        assert_eq!(
            hint("quality", &s),
            "1: 0.05, 2: 0.25, 3: 0.40, 4: 0.20, 5: 0.10"
        );
        assert_eq!(hint("severity", &s), "no answer");
        let repro = hint("reproducibility", &s);
        assert!(repro.contains("has_logs yes 0.60") && repro.contains("has_steps no answer"));

        let p = targeted_prompt("area", &issue(), &repo(), &s);
        assert!(p.contains("## Question\n\nWhich SIG"), "{p}");
        assert!(p.contains("- sig/network"));
        assert!(p.contains("Its probabilities: sig/node 0.50"));
    }

    #[test]
    fn escalated_fields_are_labeled_and_merged() {
        let note = Note {
            model: "ollama:qwen3:1.7b".into(),
            draft: decision_triage::describe_field(&draft(), "area"),
        };
        let field = note.field("sig/network".to_owned(), 85.0, "DNS is kube-proxy/CoreDNS");
        assert_eq!(field.confidence, Some(0.85));
        assert_eq!(
            field.evidence.as_deref(),
            Some(
                r#"escalated: ollama:qwen3:1.7b (decision model: "sig/node" (confidence 0.50)): DNS is kube-proxy/CoreDNS"#
            )
        );
        let mut triage = draft();
        Escalated {
            field: "area",
            overrides: Triage {
                area: Some(field),
                ..Triage::default()
            },
            clear: false,
            usage: Usage::default(),
        }
        .apply(&mut triage);
        assert_eq!(triage.area.unwrap().value, "sig/network");
        assert_eq!(triage.kind.unwrap().value, Kind::Bug);

        // An invalid answer leaves the draft alone.
        let mut triage = draft();
        Escalated {
            field: "area",
            overrides: Triage::default(),
            clear: false,
            usage: Usage::default(),
        }
        .apply(&mut triage);
        assert_eq!(triage.area.unwrap().value, "sig/node");
    }

    #[test]
    fn outdated_is_computed_from_the_reported_version() {
        let note = Note {
            model: "m".into(),
            draft: "true (confidence 0.60)".into(),
        };
        let answer = |v: &str| VersionAnswer {
            evidence: String::new(),
            kubernetes_version: v.into(),
        };
        let old = outdated(&answer("v1.32.1"), &repo(), &note).unwrap();
        assert!(old.value);
        assert!(
            old.evidence
                .unwrap()
                .ends_with("reported v1.32.1, latest v1.33.2")
        );
        assert!(!outdated(&answer("1.33.2"), &repo(), &note).unwrap().value);
        assert!(outdated(&answer(""), &repo(), &note).is_none());
        assert!(outdated(&answer("v2.1"), &repo(), &note).is_none());

        // No version stated: the draft's guess is removed.
        let mut triage = draft();
        Escalated {
            field: "outdated_version",
            overrides: Triage::default(),
            clear: true,
            usage: Usage::default(),
        }
        .apply(&mut triage);
        assert!(triage.outdated_version.is_none());
    }

    #[test]
    fn checklist_only_for_bug_like_kinds() {
        let mut t = draft();
        t.kind = Some(Field::new(Kind::Question));
        drop_checklist_unless_bug_like(&mut t);
        assert!(t.reproducibility.is_none());
        let mut t = draft();
        drop_checklist_unless_bug_like(&mut t);
        assert!(t.reproducibility.is_some());
    }

    #[test]
    fn reply_prompt_follows_the_kind() {
        let p = reply_prompt(&issue(), &repo(), &draft());
        assert!(p.contains("ask for exactly these missing details"), "{p}");
        assert!(p.contains("- the steps to reproduce the problem"));
        assert!(p.contains("- the relevant logs or error messages"));
        assert!(!p.contains("`kubectl version`"));

        let mut question = draft();
        question.kind = Some(Field::new(Kind::Question));
        question.duplicates = Some(Field::new(vec![7, 9]));
        let p = reply_prompt(&issue(), &repo(), &question);
        assert!(p.contains("discuss.kubernetes.io"));
        assert!(p.contains("#7, #9 may report the same problem"));

        let mut feature = draft();
        feature.kind = Some(Field::new(Kind::Feature));
        assert!(reply_prompt(&issue(), &repo(), &feature).contains("KEP process"));
    }

    #[test]
    fn sub_agent_asks_for_area_only_when_routed() {
        let (spec, finish) = sub_agent_spec(
            "ollama:qwen3:1.7b",
            SubAgentTask { area: true },
            &issue(),
            &repo(),
            &summary(),
        );
        assert_eq!(spec.step, SUB_AGENT_STEP);
        assert_eq!(spec.tools.len(), 2);
        assert!(spec.prompt.contains("which SIG owns the problem"));
        assert!(finish.instructions.contains("area: the SIG label"));
        let (spec, finish) = sub_agent_spec(
            "ollama:qwen3:1.7b",
            SubAgentTask { area: false },
            &issue(),
            &repo(),
            &summary(),
        );
        assert!(!spec.prompt.contains("which SIG owns"));
        assert!(
            finish
                .instructions
                .contains("area and area_confidence: null")
        );
        let schema = schemars::schema_for!(SubAgentAnswer);
        assert!(
            schema.as_value()["required"].is_null(),
            "{}",
            schema.as_value()
        );
    }

    #[test]
    fn answer_schemas_constrain_enums() {
        let schema = schemars::schema_for!(KindAnswer);
        let v = schema.as_value();
        assert_eq!(v["properties"]["kind"]["enum"][0], "bug");
    }

    #[test]
    fn answer_schemas_put_evidence_first_and_confidence_last() {
        fn props<T: JsonSchema>() -> Vec<String> {
            let schema = schemars::schema_for!(T);
            schema.as_value()["properties"]
                .as_object()
                .unwrap()
                .keys()
                .cloned()
                .collect()
        }
        for p in [
            props::<KindAnswer>(),
            props::<AreaAnswer>(),
            props::<SeverityAnswer>(),
            props::<YesNoAnswer>(),
            props::<QualityAnswer>(),
            props::<ReproAnswer>(),
            props::<ScopeAnswer>(),
        ] {
            assert_eq!(p.first().map(String::as_str), Some("evidence"), "{p:?}");
            assert_eq!(p.last().map(String::as_str), Some("confidence"), "{p:?}");
        }
        assert_eq!(props::<VersionAnswer>(), ["evidence", "kubernetes_version"]);
    }
}
