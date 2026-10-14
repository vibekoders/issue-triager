//! `SingleShot`: one LLM call with structured output fills in the whole
//! triage except duplicates, which need search.
//!
//! The model answers a flat schema with plain strings and booleans, which
//! small models follow far better than nested `Field`s; the code then maps it
//! onto `Triage`, dropping values that aren't allowed (an area that isn't one
//! of the repo's SIG labels) instead of failing the call. The outdated-version
//! check is done in code from the version the model read off the issue.

use std::fmt::Write;

use restate_sdk::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tracing::{Instrument, info};
use triage_core::config::SingleShotConfig;
use triage_core::{
    Field, Kind, QUALITY_RUBRIC, Repo, Reproducibility, Scope, Severity, Triage, TriageRequest,
    TriageResponse,
};

use super::heuristic::{K8sVersion, SIG_KEYWORDS};
use super::{parse_config, record_run, response};
use crate::models::{self, ChatRequest, Structured, journal};
use crate::trace::handler_span;

pub const VERSION: &str = "3";
const NAME: &str = "SingleShot";
const STEP: &str = "single_shot";

/// Longer bodies (mostly pasted logs) are cut so the prompt fits a small
/// local model's default context window.
const MAX_BODY_CHARS: usize = 6000;
/// A ceiling with room for a reasoning model's reasoning tokens.
const MAX_OUTPUT_TOKENS: u32 = 6000;

pub struct SingleShotStrategy;

/// Timeouts: see "Restate timeouts" in the [`super`] module docs.
#[restate_sdk::workflow(name = "SingleShot", inactivity_timeout = "5m", abort_timeout = "1h")]
impl SingleShotStrategy {
    #[handler]
    async fn run(
        &self,
        ctx: WorkflowContext<'_>,
        Json(request): Json<TriageRequest>,
    ) -> Result<Json<TriageResponse>, HandlerError> {
        let span = handler_span(NAME, ctx.key(), ctx.headers());
        async move {
            let (request, config_hash) = parse_config::<SingleShotConfig>(request)?;
            record_run(&tracing::Span::current(), VERSION, &config_hash);
            models::parse_address(&request.config.model)
                .map_err(|e| TerminalError::new_with_code(400, format!("invalid config: {e}")))?;

            let chat = chat_request(&request);
            let out: Structured<ModelTriage> =
                journal(&ctx, STEP, models::extract(STEP, chat)).await?;
            let triage = to_triage(out.value, &request.repo);
            info!(
                kind = triage.kind.as_ref().map(|k| k.value.as_str()),
                area = triage.area.as_ref().map(|a| a.value.as_str()),
                cost_usd = out.usage.cost_usd,
                "single-shot triage done"
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

/// What the model fills in. Field order is generation order, so the summary
/// comes first and gives the later answers something to lean on, and each
/// confidence follows its answer. (The schema lists properties in declaration
/// order because the workspace enables serde_json's `preserve_order`;
/// Ollama's grammar then makes the model write them in that order.)
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ModelTriage {
    /// One sentence: what the issue is about.
    pub summary: String,
    #[schemars(extend("enum" = ["bug", "feature", "question", "docs", "cleanup", "failing_test", "flake", "other", "spam"]))]
    pub kind: String,
    /// 0 to 1.
    pub kind_confidence: f64,
    /// One of the SIG labels from the prompt, or "unknown".
    pub area: String,
    /// 0 to 1.
    pub area_confidence: f64,
    #[schemars(extend("enum" = ["critical", "important_soon", "important_longterm", "backlog"]))]
    pub severity: String,
    /// 0 to 1.
    pub severity_confidence: f64,
    pub is_regression: bool,
    pub security_sensitive: bool,
    /// 1 to 5.
    pub quality: i64,
    /// The Kubernetes version the reporter runs, e.g. "v1.31.2", or "" if not stated.
    pub kubernetes_version: String,
    pub has_version: bool,
    pub has_environment: bool,
    pub has_steps: bool,
    pub has_expected_actual: bool,
    pub has_minimal_repro: bool,
    pub has_logs: bool,
    pub enough_info: bool,
    #[schemars(extend("enum" = ["single", "multiple"]))]
    pub scope: String,
    /// 0 to 1: how sure you are about the yes/no answers, quality and scope.
    pub confidence: f64,
    /// A short first reply to the author.
    pub suggested_reply: String,
}

const SYSTEM_PROMPT: &str = "You are an experienced maintainer triaging newly filed GitHub \
issues. Read the issue and fill in every field of the JSON answer, following the field guide. \
Judge only from what the issue says; do not invent facts. Confidence values are numbers from 0 \
to 1 saying how likely your answer is to be right.";

pub fn chat_request(request: &TriageRequest<SingleShotConfig>) -> ChatRequest {
    let mut chat = ChatRequest::new(&request.config.model)
        .system(SYSTEM_PROMPT)
        .user(user_prompt(request))
        .max_output_tokens(MAX_OUTPUT_TOKENS);
    if let Some(t) = request.config.temperature {
        chat = chat.temperature(f64::from(t));
    }
    chat
}

fn user_prompt(request: &TriageRequest<SingleShotConfig>) -> String {
    let issue = &request.issue;
    let repo = &request.repo;
    let issue_json = serde_json::json!({
        "number": issue.number,
        "title": issue.title,
        "author_association": issue.author_association,
        "created_at": issue.created_at,
        "body": truncate(&issue.body, MAX_BODY_CHARS),
    });
    let mut p = String::new();
    let _ = writeln!(p, "Repository: {}\n", repo.name);
    let _ = writeln!(
        p,
        "## Issue\n\n{}\n",
        serde_json::to_string_pretty(&issue_json).unwrap_or_default()
    );
    p.push_str("## Field guide\n\n");
    p.push_str("summary: one sentence saying what the issue is about.\n\n");
    p.push_str("kind, one of:\n");
    for k in Kind::ALL {
        let _ = writeln!(p, "- {}: {}", k.as_str(), k.description());
    }
    p.push_str("\narea: the SIG (special interest group) that owns the problem, one of:\n");
    for (name, hint) in sig_labels(repo) {
        let _ = writeln!(p, "- {name}: {hint}");
    }
    p.push_str("Use \"unknown\" if none fits.\n\n");
    p.push_str("severity, one of:\n");
    for s in Severity::ALL {
        let _ = writeln!(p, "- {}: {}", s.as_str(), s.description());
    }
    p.push_str(
        "\nis_regression: true if it worked in an earlier version and broke after an upgrade \
         or change.\n\
         security_sensitive: true if it reports a vulnerability or security risk (CVE, \
         privilege escalation, leaked secrets, auth bypass).\n\n",
    );
    p.push_str("quality, how well the issue is written:\n");
    for (score, text) in QUALITY_RUBRIC {
        let _ = writeln!(p, "- {score}: {text}");
    }
    let _ = writeln!(
        p,
        "\nkubernetes_version: the Kubernetes version the reporter runs (from `kubectl version` \
         output or the text), e.g. \"v1.31.2\"; \"\" if not stated.\n"
    );
    p.push_str(
        "Reproducibility checklist (true/false), judged even for non-bugs:\n\
         - has_version: states the Kubernetes or component version\n\
         - has_environment: describes the environment (cloud, OS, runtime, install method)\n\
         - has_steps: gives steps to reproduce\n\
         - has_expected_actual: says both what was expected and what happened\n\
         - has_minimal_repro: includes a minimal reproduction (manifest, commands or code)\n\
         - has_logs: includes logs, errors or stack traces\n\
         - enough_info: a maintainer could start reproducing it with what is given\n\n",
    );
    p.push_str(
        "scope: single (one problem or request) or multiple (several unrelated problems \
         bundled together).\n\n\
         confidence: how sure you are about the true/false answers, quality and scope.\n\n\
         suggested_reply: a short, friendly first reply to the author (2-4 sentences): thank \
         them, then ask for what is missing, or say what happens next.\n",
    );
    p
}

/// The repo's SIG labels with a hint about what each covers. The
/// Kubernetes label descriptions are all "Categorizes an issue or PR as
/// relevant to SIG X", so the keyword list says more when there is one.
fn sig_labels(repo: &Repo) -> Vec<(String, String)> {
    repo.labels_with_prefix("sig/")
        .map(|label| {
            let keywords =
                SIG_KEYWORDS
                    .iter()
                    .find(|(sig, _)| *sig == label.name)
                    .map(|(_, words)| {
                        words
                            .iter()
                            .take(10)
                            .copied()
                            .collect::<Vec<_>>()
                            .join(", ")
                    });
            let description = label
                .description
                .clone()
                .filter(|d| !d.is_empty() && !d.starts_with("Categorizes an issue or PR"));
            let hint = match (description, keywords) {
                (Some(d), Some(k)) => format!("{d} ({k})"),
                (Some(d), None) => d,
                (None, Some(k)) => k,
                (None, None) => String::from("-"),
            };
            (label.name.clone(), hint)
        })
        .collect()
}

/// Maps the model's answer onto `Triage`, dropping values that aren't
/// allowed.
pub fn to_triage(m: ModelTriage, repo: &Repo) -> Triage {
    let general = confidence(m.confidence);
    let kind = Kind::parse(&normalize(&m.kind));
    let area = match_sig(&m.area, repo);
    let reproducibility = match kind {
        Some(Kind::Bug | Kind::FailingTest | Kind::Flake) | None => Some(Field {
            value: Reproducibility::from_checks(
                m.has_version,
                m.has_environment,
                m.has_steps,
                m.has_expected_actual,
                m.has_minimal_repro,
                m.has_logs,
                m.enough_info,
            ),
            confidence: general,
            evidence: None,
        }),
        _ => None,
    };
    let scope = match normalize(&m.scope).as_str() {
        "single" => Some(Scope::Single),
        "multiple" => Some(Scope::Multiple),
        _ => None,
    };
    let outdated_version = outdated(&m.kubernetes_version, repo);
    Triage {
        kind: kind.map(|k| Field {
            value: k,
            confidence: confidence(m.kind_confidence),
            evidence: Some(m.summary.clone()).filter(|s| !s.is_empty()),
        }),
        area: area.map(|a| with(a, confidence(m.area_confidence))),
        severity: Severity::parse(&normalize(&m.severity))
            .map(|s| with(s, confidence(m.severity_confidence))),
        is_regression: Some(with(m.is_regression, general)),
        security_sensitive: Some(with(m.security_sensitive, general)),
        quality: (1..=5)
            .contains(&m.quality)
            .then(|| with(m.quality as u8, general)),
        reproducibility,
        outdated_version,
        scope: scope.map(|s| with(s, general)),
        duplicates: None,
        suggested_reply: Some(m.suggested_reply.trim().to_owned())
            .filter(|r| !r.is_empty())
            .map(Field::new),
    }
}

fn with<T>(value: T, confidence: Option<f64>) -> Field<T> {
    Field {
        value,
        confidence,
        evidence: None,
    }
}

fn normalize(s: &str) -> String {
    s.trim().to_lowercase().replace([' ', '-'], "_")
}

/// Accepts `sig/node`, `node`, `SIG Node`; anything else is dropped.
fn match_sig(answer: &str, repo: &Repo) -> Option<String> {
    let wanted = answer
        .trim()
        .to_lowercase()
        .replace("sig-", "sig/")
        .replace("sig ", "sig/");
    let wanted = wanted
        .strip_prefix("sig/")
        .unwrap_or(&wanted)
        .replace(' ', "-");
    repo.labels_with_prefix("sig/")
        .find(|l| l.name["sig/".len()..].eq_ignore_ascii_case(&wanted))
        .map(|l| l.name.clone())
}

/// Clamps a self-reported confidence into 0..=1, reading values above 1 as
/// percentages, which small models sometimes give.
fn confidence(c: f64) -> Option<f64> {
    if !c.is_finite() || c < 0.0 {
        return None;
    }
    Some(if c > 1.0 { (c / 100.0).min(1.0) } else { c })
}

fn outdated(reported: &str, repo: &Repo) -> Option<Field<bool>> {
    let latest_tag = repo.latest_release.as_deref()?;
    let latest = K8sVersion::parse(latest_tag)?;
    let version = K8sVersion::parse(reported)?;
    (version.major == 1).then(|| {
        Field::new(version.older_than(&latest))
            .with_evidence(format!("reported {}, latest {latest_tag}", reported.trim()))
    })
}

fn truncate(text: &str, max_chars: usize) -> String {
    match text.char_indices().nth(max_chars) {
        Some((i, _)) => format!("{}\n[... truncated]", &text[..i]),
        None => text.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use triage_core::{Issue, Label};

    fn repo() -> Repo {
        Repo {
            name: "kubernetes/kubernetes".into(),
            labels: vec![
                Label {
                    name: "sig/node".into(),
                    description: Some("Categorizes an issue or PR as relevant to SIG Node.".into()),
                },
                Label {
                    name: "sig/api-machinery".into(),
                    description: None,
                },
                Label {
                    name: "kind/bug".into(),
                    description: None,
                },
            ],
            latest_release: Some("v1.33.2".into()),
            as_of: "2025-07-01T00:00:00Z".into(),
        }
    }

    fn answer() -> ModelTriage {
        serde_json::from_value(serde_json::json!({
            "summary": "kubelet restarts pods",
            "kind": "Bug", "kind_confidence": 0.9,
            "area": "SIG Node", "area_confidence": 80,
            "severity": "important-soon", "severity_confidence": 0.5,
            "is_regression": true, "security_sensitive": false, "quality": 4,
            "kubernetes_version": "v1.32.1",
            "has_version": true, "has_environment": false, "has_steps": true,
            "has_expected_actual": true, "has_minimal_repro": false, "has_logs": true,
            "enough_info": true, "scope": "single", "confidence": 0.7,
            "suggested_reply": " Thanks! "
        }))
        .unwrap()
    }

    #[test]
    fn maps_and_normalizes() {
        let t = to_triage(answer(), &repo());
        assert_eq!(t.kind.unwrap().value, Kind::Bug);
        let area = t.area.unwrap();
        assert_eq!(area.value, "sig/node");
        assert_eq!(area.confidence, Some(0.8));
        assert_eq!(t.severity.unwrap().value, Severity::ImportantSoon);
        assert!(t.outdated_version.unwrap().value);
        let repro = t.reproducibility.unwrap().value;
        assert_eq!(repro.missing.len(), 2);
        assert_eq!(t.suggested_reply.unwrap().value, "Thanks!");
        assert!(t.duplicates.is_none());
    }

    #[test]
    fn drops_invalid_values() {
        let mut a = answer();
        a.area = "sig/windows".into();
        a.kind = "enhancement".into();
        a.quality = 9;
        a.kubernetes_version = String::new();
        let t = to_triage(a, &repo());
        assert!(t.area.is_none() && t.kind.is_none() && t.quality.is_none());
        assert!(t.outdated_version.is_none());
        // Unknown kind: the checklist is still kept.
        assert!(t.reproducibility.is_some());
    }

    #[test]
    fn prompt_lists_sigs_with_hints() {
        let request = TriageRequest {
            config: SingleShotConfig {
                model: "ollama:qwen3:1.7b".into(),
                temperature: None,
            },
            issue: Issue {
                number: 1,
                title: "t".into(),
                body: "b".repeat(MAX_BODY_CHARS + 10),
                author_association: "NONE".into(),
                created_at: "2025-07-01T00:00:00Z".into(),
            },
            repo: repo(),
        };
        let p = user_prompt(&request);
        assert!(p.contains("- sig/node: kubelet, pod"), "{p}");
        assert!(p.contains("- sig/api-machinery: apiserver"));
        assert!(!p.contains("kind/bug"));
        assert!(p.contains("[... truncated]"));
        assert!(p.contains("5: Exemplary"));
    }

    #[test]
    fn schema_constrains_enums() {
        let schema = schemars::schema_for!(ModelTriage);
        let kind = &schema.as_value()["properties"]["kind"];
        assert_eq!(kind["type"], "string");
        assert_eq!(kind["enum"][0], "bug");
    }

    #[test]
    fn schema_lists_fields_in_generation_order() {
        let schema = schemars::schema_for!(ModelTriage);
        let props: Vec<&str> = schema.as_value()["properties"]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(props.first(), Some(&"summary"));
        assert_eq!(&props[1..3], ["kind", "kind_confidence"]);
        assert_eq!(props.last(), Some(&"suggested_reply"));
    }
}
