//! `DraftVerify`: the decision model's draft is stored in workflow state and
//! is readable early through a shared `preliminary()` handler; then one agent
//! checks only the fields below the confidence threshold.
//!
//! Config: `decision_model`, `verifier_model`, `thresholds`, `max_turns`.
//! Compared against `Workflow` with escalation: workflow-coordinated versus
//! agent-coordinated escalation.
//!
//! Steps:
//! 1. One decision call drafts every typed field ([`decision_triage`]). The
//!    draft, with its usage, is stored as `preliminary`, so a caller can read
//!    it (`/DraftVerify/<key>/preliminary`) long before the agent is done.
//! 2. The fields below their threshold are picked in code, leaving out those
//!    a verifier couldn't fill in or change (a confident feature request has
//!    no checklist; without a latest release there is no `outdated_version`):
//!    [`decision_triage::escalatable_fields`], the same routing as `Workflow`.
//! 3. If there are none, the draft is the answer. Otherwise the agent loop
//!    runs once, with the issue, the draft and the fields to verify; its
//!    final answer covers only those fields (plus duplicates, which a search
//!    may turn up) and is merged over the draft.
//!
//! Every verified field carries the evidence `verified by agent` (with the
//! draft value when the agent changed it). Its confidence is cleared: an
//! LLM's self-reported confidence isn't comparable with the decision
//! model's probabilities.

use std::fmt::Write;

use restate_sdk::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tracing::{Instrument, info};
use triage_core::config::DraftVerifyConfig;
use triage_core::{
    Field, Issue, Kind, QUALITY_RUBRIC, Repo, Reproducibility, Scope, Severity, Triage,
    TriageRequest, TriageResponse,
};

use super::agent::{check_address, issue_prompt, valid_duplicates};
use super::agent_loop::{self, AgentOutcome, AgentSpec, Finish};
use super::decision_triage::{self, bug_like, describe_field, escalatable_fields, match_sig};
use super::{parse_config, record_run, response};
use crate::tools::{self, ToolContext};
use crate::trace::handler_span;

pub const VERSION: &str = "4";
const NAME: &str = "DraftVerify";
const STEP: &str = "verify";

/// The workflow state key of the draft response.
pub const PRELIMINARY: &str = "preliminary";

const MAX_BODY_CHARS: usize = 3000;

pub struct DraftVerifyStrategy;

/// Timeouts: see "Restate timeouts" in the [`super`] module docs.
#[restate_sdk::workflow(name = "DraftVerify", inactivity_timeout = "5m", abort_timeout = "1h")]
impl DraftVerifyStrategy {
    #[handler]
    async fn run(
        &self,
        ctx: WorkflowContext<'_>,
        Json(request): Json<TriageRequest>,
    ) -> Result<Json<TriageResponse>, HandlerError> {
        let span = handler_span(NAME, ctx.key(), ctx.headers());
        async move {
            let (request, config_hash) = parse_config::<DraftVerifyConfig>(request)?;
            record_run(&tracing::Span::current(), VERSION, &config_hash);
            let config = &request.config;
            check_address(&config.decision_model)?;
            check_address(&config.verifier_model)?;
            let (issue, repo) = (&request.issue, &request.repo);

            let (draft, summary) =
                decision_triage::run(&ctx, &config.decision_model, issue, repo).await?;
            let mut usage = summary.usage.clone();
            let preliminary = response(
                NAME,
                VERSION,
                config_hash.clone(),
                draft.clone(),
                usage.clone(),
            );
            ctx.set(PRELIMINARY, Json(preliminary.clone()));

            let fields = escalatable_fields(&draft, &config.thresholds, repo);
            if fields.is_empty() {
                info!("draft accepted: every field a verifier could change is above its threshold");
                return Ok(Json(preliminary));
            }

            let tool_ctx = ToolContext::new(repo.clone(), issue.number);
            let prompt = verify_prompt(issue, repo, &draft, &fields);
            let offered = verifier_tools(&fields);
            let spec = AgentSpec::new(
                STEP,
                &config.verifier_model,
                system_prompt(&offered),
                prompt.clone(),
            )
            .tools(tools::definitions_for(&offered))
            .max_turns(config.max_turns);
            let finish = Finish::new(FINAL_SYSTEM_PROMPT, prompt, final_instructions(&fields));
            let out: AgentOutcome<Verified> =
                agent_loop::run(&ctx, &tool_ctx, &spec, &finish).await?;
            usage += &out.usage;

            let checked = apply(&draft, &out.answer, &fields, repo);
            let mut triage = decision_triage::merge(draft, checked.overrides);
            if out.searched() {
                triage.duplicates = Some(
                    Field::new(valid_duplicates(
                        &out.answer.duplicates,
                        &out.found_issues,
                        issue.number,
                    ))
                    .with_evidence("found by the verifier's search"),
                );
            }
            // Like the other strategies: no checklist for kinds that aren't
            // bug-like, which verifying `kind` may have turned it into.
            if !bug_like(&triage) {
                triage.reproducibility = None;
            }
            info!(
                verify.fields = fields.join(","),
                verify.changed = checked.changed.join(","),
                verify.unanswered = checked.unanswered.join(","),
                turns = out.turns,
                cost_usd = usage.cost_usd,
                "draft verified"
            );
            Ok(Json(response(NAME, VERSION, config_hash, triage, usage)))
        }
        .instrument(span)
        .await
    }

    /// The decision model's draft, as soon as it exists; `None` (an empty
    /// body) before that.
    #[handler]
    async fn preliminary(
        &self,
        ctx: SharedWorkflowContext<'_>,
    ) -> Result<Option<Json<TriageResponse>>, HandlerError> {
        Ok(ctx.get::<Json<TriageResponse>>(PRELIMINARY).await?)
    }
}

/// The tools that can help with `fields`. Search is always offered (it
/// finds duplicates, and earlier reports show the area); the reference
/// lookups only when a field needs them, since a small model tends to call
/// every tool it is given.
fn verifier_tools(fields: &[&str]) -> Vec<&'static str> {
    let mut names = vec![tools::SEARCH_ISSUES, tools::GET_ISSUE];
    if fields.contains(&"outdated_version") {
        names.push(tools::GET_RELEASES);
    }
    if fields.contains(&"reproducibility") || fields.contains(&"quality") {
        names.push(tools::GET_ISSUE_TEMPLATES);
    }
    names
}

fn system_prompt(tools_offered: &[&str]) -> String {
    let mut p = String::from(
        "You are an experienced maintainer checking a drafted triage of a newly filed GitHub \
         issue. A fast classifier drafted it; some of its answers are uncertain. Check only the \
         fields you are asked to verify. Use the tools where they help:\n\
         - search_issues and get_issue: find earlier reports of the same problem (possible \
         duplicates), and see which area such problems belong to.\n",
    );
    if tools_offered.contains(&tools::GET_RELEASES) {
        p.push_str("- get_releases: see whether the reported version is current.\n");
    }
    if tools_offered.contains(&tools::GET_ISSUE_TEMPLATES) {
        p.push_str("- get_issue_templates: see what a complete report contains.\n");
    }
    p.push_str(
        "Make one or two targeted calls. When you are done, reply without calling a tool, \
         giving your verdict for each field to verify.",
    );
    p
}

const FINAL_SYSTEM_PROMPT: &str = "You are an experienced maintainer checking a drafted triage \
of a GitHub issue. Answer in JSON. Fill in only the fields listed under \"Fields to verify\", \
from the issue and the findings, and leave every other field null. Judge only from what the \
issue and the findings say; do not invent facts.";

/// The issue, the draft with confidences, and a guide to each field to
/// verify. Both the agent and the final call read it.
fn verify_prompt(issue: &Issue, repo: &Repo, draft: &Triage, fields: &[&str]) -> String {
    let mut p = issue_prompt(issue, repo, MAX_BODY_CHARS);
    p.push_str("\n## Draft triage\n\n");
    for name in decision_triage::FIELDS {
        let _ = writeln!(p, "- {name}: {}", describe_field(draft, name));
    }
    p.push_str("\n## Fields to verify\n\n");
    for name in fields {
        let _ = writeln!(p, "{name}: {}\n", field_guide(name, issue, repo));
    }
    p
}

fn field_guide(name: &str, issue: &Issue, repo: &Repo) -> String {
    let mut g = String::new();
    match name {
        "kind" => {
            g.push_str("one of:\n");
            for k in Kind::ALL {
                let _ = writeln!(g, "- {}: {}", k.as_str(), k.description());
            }
        }
        "area" => {
            g.push_str("the SIG that owns the problem, one of:\n");
            for (sig, hint) in decision_triage::sig_options(issue, repo) {
                let _ = writeln!(g, "- {sig}: {hint}");
            }
        }
        "severity" => {
            g.push_str("one of:\n");
            for s in Severity::ALL {
                let _ = writeln!(g, "- {}: {}", s.as_str(), s.description());
            }
        }
        "is_regression" => g.push_str(
            "true if it worked in an earlier version and broke after an upgrade or change.",
        ),
        "security_sensitive" => g.push_str(
            "true if it reports a vulnerability or security risk (CVE, privilege escalation, \
             leaked secrets, auth bypass).",
        ),
        "quality" => {
            g.push_str("how well the issue is written, 1 to 5:\n");
            for (score, text) in QUALITY_RUBRIC {
                let _ = writeln!(g, "- {score}: {text}");
            }
        }
        "reproducibility" => g.push_str(
            "the checklist, each true or false: has_version (states the version), \
             has_environment (cloud, OS, runtime, install method), has_steps (steps to \
             reproduce), has_expected_actual (both expected and actual behavior), \
             has_minimal_repro (a manifest, commands or code), has_logs (logs, errors or stack \
             traces), and enough_info (a maintainer could start reproducing it).",
        ),
        "outdated_version" => {
            let _ = write!(
                g,
                "true if the Kubernetes version the reporter runs is older than the latest \
                 release when the issue was filed ({}); null if the issue states no version.",
                repo.latest_release.as_deref().unwrap_or("unknown")
            );
        }
        "scope" => g.push_str(
            "single (one problem or request) or multiple (several unrelated problems bundled \
             together).",
        ),
        _ => {}
    }
    g.trim_end().to_owned()
}

fn final_instructions(fields: &[&str]) -> String {
    let names: Vec<&str> = fields
        .iter()
        .flat_map(|&f| match f {
            "reproducibility" => vec![
                "has_version",
                "has_environment",
                "has_steps",
                "has_expected_actual",
                "has_minimal_repro",
                "has_logs",
                "enough_info",
            ],
            other => vec![other],
        })
        .collect();
    format!(
        "Fill in only these fields: {}. Give a value for each of them: the corrected value \
         where the draft is wrong, the draft's value where it is right. Leave all other fields \
         null.\n\
         duplicates: the numbers of earlier issues from the findings that report the same \
         problem as this issue, most likely first; [] if none does or nothing was searched.",
        names.join(", ")
    )
}

/// The verifier's answer. Every field is optional: only the ones being
/// verified are asked for, and anything else it fills in is ignored.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct Verified {
    /// bug, feature, question, docs, cleanup, failing_test, flake, other or spam.
    #[serde(default)]
    pub kind: Option<String>,
    /// One of the SIG labels, e.g. "sig/node".
    #[serde(default)]
    pub area: Option<String>,
    /// critical, important_soon, important_longterm or backlog.
    #[serde(default)]
    pub severity: Option<String>,
    #[serde(default)]
    pub is_regression: Option<bool>,
    #[serde(default)]
    pub security_sensitive: Option<bool>,
    /// 1 to 5.
    #[serde(default)]
    pub quality: Option<i64>,
    #[serde(default)]
    pub has_version: Option<bool>,
    #[serde(default)]
    pub has_environment: Option<bool>,
    #[serde(default)]
    pub has_steps: Option<bool>,
    #[serde(default)]
    pub has_expected_actual: Option<bool>,
    #[serde(default)]
    pub has_minimal_repro: Option<bool>,
    #[serde(default)]
    pub has_logs: Option<bool>,
    #[serde(default)]
    pub enough_info: Option<bool>,
    #[serde(default)]
    pub outdated_version: Option<bool>,
    /// single or multiple.
    #[serde(default)]
    pub scope: Option<String>,
    /// Earlier issues from the findings that report the same problem.
    #[serde(default)]
    pub duplicates: Vec<u64>,
}

/// The verified fields as a partial `Triage`, and what happened to each.
struct Checked {
    overrides: Triage,
    /// Verified fields whose value differs from the draft.
    changed: Vec<&'static str>,
    /// Fields to verify the agent gave no valid answer for (the draft stays).
    unanswered: Vec<&'static str>,
}

/// Takes the verifier's answers for `fields` only, dropping invalid values
/// (an area that isn't a SIG label, a quality of 9).
fn apply(draft: &Triage, v: &Verified, fields: &[&'static str], repo: &Repo) -> Checked {
    let mut out = Triage::default();
    let mut changed = Vec::new();
    let mut unanswered = Vec::new();
    for &name in fields {
        let answered = match name {
            "kind" => set(
                &mut out.kind,
                &draft.kind,
                v.kind.as_deref().and_then(|k| Kind::parse(&normalize(k))),
            ),
            "area" => set(
                &mut out.area,
                &draft.area,
                v.area.as_deref().and_then(|a| match_sig(a, repo)),
            ),
            "severity" => set(
                &mut out.severity,
                &draft.severity,
                v.severity
                    .as_deref()
                    .and_then(|s| Severity::parse(&normalize(s))),
            ),
            "is_regression" => set(
                &mut out.is_regression,
                &draft.is_regression,
                v.is_regression,
            ),
            "security_sensitive" => set(
                &mut out.security_sensitive,
                &draft.security_sensitive,
                v.security_sensitive,
            ),
            "quality" => set(
                &mut out.quality,
                &draft.quality,
                v.quality.filter(|q| (1..=5).contains(q)).map(|q| q as u8),
            ),
            "reproducibility" => set(
                &mut out.reproducibility,
                &draft.reproducibility,
                reproducibility(draft, v),
            ),
            "outdated_version" => set(
                &mut out.outdated_version,
                &draft.outdated_version,
                v.outdated_version,
            ),
            "scope" => set(
                &mut out.scope,
                &draft.scope,
                v.scope
                    .as_deref()
                    .and_then(|s| match normalize(s).as_str() {
                        "single" => Some(Scope::Single),
                        "multiple" => Some(Scope::Multiple),
                        _ => None,
                    }),
            ),
            _ => None,
        };
        match answered {
            None => unanswered.push(name),
            Some(true) => changed.push(name),
            Some(false) => {}
        }
    }
    Checked {
        overrides: out,
        changed,
        unanswered,
    }
}

/// Puts the verified value into `slot` with its evidence. Returns whether
/// it changed the draft, or `None` when there was no valid answer.
fn set<T: PartialEq + Serialize>(
    slot: &mut Option<Field<T>>,
    draft: &Option<Field<T>>,
    value: Option<T>,
) -> Option<bool> {
    let value = value?;
    let changed = draft.as_ref().is_none_or(|d| d.value != value);
    let evidence = match draft {
        Some(d) if changed => format!(
            "verified by agent: changed from {} ({})",
            serde_json::to_string(&d.value).unwrap_or_default(),
            d.confidence
                .map(|c| format!("confidence {c:.2}"))
                .unwrap_or_else(|| "no confidence".into())
        ),
        None => "verified by agent: the draft had no answer".to_owned(),
        Some(_) => "verified by agent: confirmed the draft".to_owned(),
    };
    *slot = Some(Field::new(value).with_evidence(evidence));
    Some(changed)
}

/// The draft's checklist with the verifier's items over it; `None` when the
/// verifier answered none of them.
fn reproducibility(draft: &Triage, v: &Verified) -> Option<Reproducibility> {
    let items = [
        v.has_version,
        v.has_environment,
        v.has_steps,
        v.has_expected_actual,
        v.has_minimal_repro,
        v.has_logs,
        v.enough_info,
    ];
    if items.iter().all(Option::is_none) {
        return None;
    }
    let base = draft
        .reproducibility
        .as_ref()
        .map(|f| f.value.clone())
        .unwrap_or_default();
    Some(Reproducibility::from_checks(
        v.has_version.unwrap_or(base.has_version),
        v.has_environment.unwrap_or(base.has_environment),
        v.has_steps.unwrap_or(base.has_steps),
        v.has_expected_actual.unwrap_or(base.has_expected_actual),
        v.has_minimal_repro.unwrap_or(base.has_minimal_repro),
        v.has_logs.unwrap_or(base.has_logs),
        v.enough_info.unwrap_or(base.enough_info),
    ))
}

fn normalize(s: &str) -> String {
    s.trim().to_lowercase().replace([' ', '-'], "_")
}

#[cfg(test)]
mod tests {
    use super::*;
    use triage_core::Label;

    fn repo() -> Repo {
        Repo {
            name: "kubernetes/kubernetes".into(),
            labels: ["sig/node", "sig/network"]
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
            body: "After upgrading, pods can't resolve services.".into(),
            author_association: "NONE".into(),
            created_at: "2025-07-01T00:00:00Z".into(),
        }
    }

    fn draft() -> Triage {
        Triage {
            kind: Some(Field::new(Kind::Bug).with_confidence(0.95)),
            area: Some(Field::new("sig/node".to_owned()).with_confidence(0.4)),
            quality: Some(Field::new(3).with_confidence(0.5)),
            reproducibility: Some(
                Field::new(Reproducibility::from_checks(
                    true, false, false, true, false, false, false,
                ))
                .with_confidence(0.6),
            ),
            scope: Some(Field::new(Scope::Single).with_confidence(0.6)),
            ..Triage::default()
        }
    }

    #[test]
    fn applies_only_the_fields_to_verify() {
        let v = Verified {
            kind: Some("feature".into()),
            area: Some("SIG Network".into()),
            quality: Some(3),
            has_steps: Some(true),
            enough_info: Some(true),
            scope: Some("bogus".into()),
            ..Verified::default()
        };
        let fields = ["area", "quality", "reproducibility", "scope", "severity"];
        let checked = apply(&draft(), &v, &fields, &repo());
        assert_eq!(checked.changed, vec!["area", "reproducibility"]);
        assert_eq!(checked.unanswered, vec!["scope", "severity"]);
        let o = &checked.overrides;
        // `kind` wasn't asked for, so the verifier's answer is ignored.
        assert!(o.kind.is_none());
        let area = o.area.as_ref().unwrap();
        assert_eq!(area.value, "sig/network");
        assert_eq!(area.confidence, None);
        assert_eq!(
            area.evidence.as_deref(),
            Some(r#"verified by agent: changed from "sig/node" (confidence 0.40)"#)
        );
        assert_eq!(
            o.quality.as_ref().unwrap().evidence.as_deref(),
            Some("verified by agent: confirmed the draft")
        );
        let repro = &o.reproducibility.as_ref().unwrap().value;
        assert!(repro.has_version && repro.has_steps && repro.enough_info);
        assert!(!repro.has_logs);

        let merged = decision_triage::merge(draft(), checked.overrides);
        assert_eq!(merged.kind.unwrap().value, Kind::Bug);
        assert_eq!(merged.scope.unwrap().confidence, Some(0.6));
    }

    #[test]
    fn prompt_shows_draft_and_guides() {
        let fields = ["area", "reproducibility", "outdated_version"];
        let p = verify_prompt(&issue(), &repo(), &draft(), &fields);
        assert!(p.contains(r#"- area: "sig/node" (confidence 0.40)"#), "{p}");
        assert!(p.contains("- severity: missing"));
        assert!(p.contains("## Fields to verify\n\narea: the SIG"));
        assert!(p.contains("- sig/network"));
        assert!(p.contains("(v1.33.2)"));
        let i = final_instructions(&fields);
        assert!(
            i.starts_with("Fill in only these fields: area, has_version,"),
            "{i}"
        );
        assert!(i.contains("enough_info, outdated_version."));
    }

    #[test]
    fn tools_follow_the_fields() {
        assert_eq!(
            verifier_tools(&["area"]),
            vec![tools::SEARCH_ISSUES, tools::GET_ISSUE]
        );
        let all = verifier_tools(&["outdated_version", "quality"]);
        assert_eq!(all.len(), 4);
        let p = system_prompt(&all);
        assert!(p.contains("- get_releases") && p.contains("- get_issue_templates"));
        assert!(!system_prompt(&verifier_tools(&["kind"])).contains("get_releases"));
    }

    #[test]
    fn verified_schema_allows_nulls() {
        let v: Verified = serde_json::from_str(r#"{"area": "sig/node"}"#).unwrap();
        assert_eq!(v.area.as_deref(), Some("sig/node"));
        assert!(v.duplicates.is_empty());
        let schema = schemars::schema_for!(Verified);
        assert!(
            schema.as_value()["required"].is_null(),
            "{}",
            schema.as_value()
        );
    }
}
