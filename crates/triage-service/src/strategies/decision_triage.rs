//! One decision-model call that answers every typed `Triage` field over the
//! issue, shared by `Workflow` and `DraftVerify`.
//!
//! The model reads the issue as JSON (`{issue: {title, body,
//! author_association, created_at}, repo: {latest_release}}`) and answers one
//! named question per field: choices for kind, area and severity, a score for
//! quality, and yes/no for the rest, including each reproducibility checklist
//! item. Decision models return probabilities rather than text, so every
//! field comes back with a confidence the caller can route on:
//!
//! ```ignore
//! let (draft, summary) = decision_triage::run(&ctx, &model, &issue, &repo).await?;
//! let unsure = decision_triage::escalatable_fields(&draft, &thresholds, &repo);
//! // ...escalate `unsure`, then:
//! let triage = decision_triage::merge(draft, escalated);
//! ```
//!
//! The request stays within every System One provider's limits (Ollama: 64
//! questions, 26 options, 26 levels; TypeSafe: 255 options, 10 levels). A
//! small model's context window may still be too short for the issue plus
//! fifteen questions; [`journaled`] then splits the questions into a few calls,
//! each journaled on its own.

use std::fmt::Write;

use allama::decision::DecisionInput;
use restate_sdk::prelude::*;
use serde::Serialize;
use triage_core::config::Thresholds;
use triage_core::{
    Field, Issue, Kind, QUALITY_RUBRIC, Repo, Reproducibility, Scope, Severity, Triage,
};

use super::heuristic::SIG_KEYWORDS;
use crate::models::{
    self, CallError, DecisionAnswer, DecisionRequest, DecisionSummary, Question, journal,
};

/// The `llm.step` of the decision call, and its journal entry name.
pub const STEP: &str = "decision_triage";

/// The most options a choice may have on every provider (Ollama's limit).
pub const MAX_CHOICE_OPTIONS: usize = 26;

/// Body lengths to try, longest first. A System One model reads one prompt
/// per question, each holding the input and every question of the request,
/// and each must fit its context window. Characters are only a rough proxy
/// for tokens (pasted logs and YAML tokenize densely), so [`decide_group`] steps
/// down when the provider reports the context length exceeded.
pub const BODY_BUDGETS: [usize; 4] = [3000, 2000, 1000, 400];
/// Keywords per SIG in the area options: enough to tell the SIGs apart,
/// few enough that the 26 options stay a few hundred tokens.
const SIG_HINT_KEYWORDS: usize = 5;

/// The typed fields the decision call answers, in `Triage` order. These are
/// the fields [`low_confidence_fields`] checks; `duplicates` and
/// `suggested_reply` need search or writing, which a decision model can't do.
pub const FIELDS: [&str; 9] = [
    "kind",
    "area",
    "severity",
    "is_regression",
    "security_sensitive",
    "quality",
    "reproducibility",
    "outdated_version",
    "scope",
];

/// Question names. Most are the `Triage` field they answer; `multiple_problems`
/// answers `scope`, and the checklist items plus `enough_info` together
/// answer `reproducibility`.
pub mod q {
    pub const KIND: &str = "kind";
    pub const AREA: &str = "area";
    pub const SEVERITY: &str = "severity";
    pub const IS_REGRESSION: &str = "is_regression";
    pub const SECURITY_SENSITIVE: &str = "security_sensitive";
    pub const MULTIPLE_PROBLEMS: &str = "multiple_problems";
    pub const OUTDATED_VERSION: &str = "outdated_version";
    pub const QUALITY: &str = "quality";
    pub const HAS_VERSION: &str = "has_version";
    pub const HAS_ENVIRONMENT: &str = "has_environment";
    pub const HAS_STEPS: &str = "has_steps";
    pub const HAS_EXPECTED_ACTUAL: &str = "has_expected_actual";
    pub const HAS_MINIMAL_REPRO: &str = "has_minimal_repro";
    pub const HAS_LOGS: &str = "has_logs";
    pub const ENOUGH_INFO: &str = "enough_info";
}

/// The reproducibility checklist questions, in `Reproducibility` order.
const CHECKLIST: [(&str, &str); 6] = [
    (
        q::HAS_VERSION,
        "Does the issue state the Kubernetes or component version the reporter runs?",
    ),
    (
        q::HAS_ENVIRONMENT,
        "Does the issue describe the environment (cloud provider, OS, container runtime, install method)?",
    ),
    (
        q::HAS_STEPS,
        "Does the issue give steps to reproduce the problem?",
    ),
    (
        q::HAS_EXPECTED_ACTUAL,
        "Does the issue say both what was expected and what actually happened?",
    ),
    (
        q::HAS_MINIMAL_REPRO,
        "Does the issue include a minimal reproduction (a manifest, commands or code)?",
    ),
    (
        q::HAS_LOGS,
        "Does the issue include logs, error messages or stack traces?",
    ),
];

/// The decision model's input. A struct rather than `json!`, because the
/// model reads the keys in order and `serde_json::Value` sorts them.
#[derive(Serialize)]
struct State<'a> {
    issue: StateIssue<'a>,
    repo: StateRepo<'a>,
}

#[derive(Serialize)]
struct StateIssue<'a> {
    title: &'a str,
    body: String,
    author_association: &'a str,
    created_at: &'a str,
}

#[derive(Serialize)]
struct StateRepo<'a> {
    latest_release: Option<&'a str>,
}

/// The issue as the decision model reads it, the body cut at the first of
/// [`BODY_BUDGETS`]. Also the input of the agent loop's `decide` tool, so
/// its answers are about the same document.
pub fn input(issue: &Issue, repo: &Repo) -> DecisionInput {
    input_with_body_limit(issue, repo, BODY_BUDGETS[0])
}

pub fn input_with_body_limit(issue: &Issue, repo: &Repo, max_body_chars: usize) -> DecisionInput {
    let state = State {
        issue: StateIssue {
            title: &issue.title,
            body: truncate(&issue.body, max_body_chars),
            author_association: &issue.author_association,
            created_at: &issue.created_at,
        },
        repo: StateRepo {
            latest_release: repo.latest_release.as_deref(),
        },
    };
    // Only fails for scalars; a struct with string fields always serializes
    // to an object.
    DecisionInput::json(&state).expect("the state serializes to a JSON object")
}

/// The decision request answering every typed field. `model` is the
/// decision model's address, e.g. `ollama:tev1:0.8b-q8_0`. The body is cut
/// at the first of [`BODY_BUDGETS`].
pub fn request(model: &str, issue: &Issue, repo: &Repo) -> DecisionRequest {
    request_with_body_limit(model, issue, repo, BODY_BUDGETS[0])
}

pub fn request_with_body_limit(
    model: &str,
    issue: &Issue,
    repo: &Repo,
    max_body_chars: usize,
) -> DecisionRequest {
    let mut kind = Question::choice("What kind of GitHub issue is this?");
    for k in Kind::ALL {
        kind = kind.option(k.as_str(), k.description());
    }
    let mut severity =
        Question::choice("How urgently does this issue need to be addressed by the maintainers?");
    for s in Severity::ALL {
        severity = severity.option(s.as_str(), s.description());
    }
    let mut quality = Question::score(
        "How well is this issue written: clear, complete and actionable for a maintainer?",
    );
    for (score, text) in QUALITY_RUBRIC {
        quality = quality.level(score.to_string(), text);
    }

    let mut request =
        DecisionRequest::new(model, input_with_body_limit(issue, repo, max_body_chars))
            .question(q::KIND, kind);
    let sigs = sig_options(issue, repo);
    // A choice needs at least two options; with fewer the repo has nothing
    // to choose between.
    if sigs.len() >= 2 {
        let mut area = Question::choice(
            "Which SIG (special interest group) owns the problem or request in this issue?",
        );
        for (name, description) in sigs {
            area = area.option(name, description);
        }
        request = request.question(q::AREA, area);
    }
    request = request
        .question(q::SEVERITY, severity)
        .question(
            q::IS_REGRESSION,
            Question::yes_no(
                "Is this a regression: did it work in an earlier version and break after an upgrade or change?",
            ),
        )
        .question(
            q::SECURITY_SENSITIVE,
            Question::yes_no(
                "Does the issue report a vulnerability or security risk (CVE, privilege escalation, leaked secrets, auth bypass)?",
            ),
        )
        .question(
            q::MULTIPLE_PROBLEMS,
            Question::yes_no(
                "Does the issue bundle several unrelated problems or requests, rather than one?",
            ),
        );
    if let Some(latest) = &repo.latest_release {
        request = request.question(
            q::OUTDATED_VERSION,
            Question::yes_no(format!(
                "Is the Kubernetes version the reporter runs older than the latest release, {latest} (repo.latest_release)?"
            )),
        );
    }
    request = request.question(q::QUALITY, quality);
    for (name, text) in CHECKLIST {
        request = request.question(name, Question::yes_no(text));
    }
    request.question(
        q::ENOUGH_INFO,
        Question::yes_no(
            "Could a maintainer start reproducing or acting on the issue with the information given?",
        ),
    )
}

/// The repo's SIG labels as choice options, with a description of what each
/// covers. The Kubernetes descriptions are all "Categorizes an issue or PR as
/// relevant to SIG X", which says nothing the key doesn't, so those are
/// replaced by the first few of the heuristic's keywords. When the repo has
/// more SIGs than a choice may have options, the ones whose keywords the
/// issue mentions most are kept.
pub fn sig_options(issue: &Issue, repo: &Repo) -> Vec<(String, String)> {
    let mut sigs: Vec<(String, String)> = repo
        .labels_with_prefix("sig/")
        .map(|label| {
            let keywords = sig_keywords(&label.name).map(|words| {
                words
                    .iter()
                    .take(SIG_HINT_KEYWORDS)
                    .copied()
                    .collect::<Vec<_>>()
                    .join(", ")
            });
            let description = label
                .description
                .as_deref()
                .map(str::trim)
                .filter(|d| !d.is_empty() && !d.starts_with("Categorizes an issue or PR"))
                .map(|d| d.trim_end_matches('.').to_owned());
            let text = match (description, keywords) {
                (Some(d), Some(k)) => format!("{d}: {k}"),
                (Some(d), None) => d,
                (None, Some(k)) => k,
                (None, None) => format!("Owned by {}", label.name),
            };
            (label.name.clone(), text)
        })
        .collect();
    if sigs.len() > MAX_CHOICE_OPTIONS {
        let text = format!("{}\n{}", issue.title, issue.body).to_lowercase();
        let hits = |sig: &str| -> usize {
            sig_keywords(sig)
                .map(|words| words.iter().map(|w| text.matches(w).count().min(3)).sum())
                .unwrap_or(0)
        };
        // Stable, so ties keep the repo's label order.
        sigs.sort_by_key(|(name, _)| std::cmp::Reverse(hits(name)));
        sigs.truncate(MAX_CHOICE_OPTIONS);
    }
    sigs
}

fn sig_keywords(sig: &str) -> Option<&'static [&'static str]> {
    SIG_KEYWORDS
        .iter()
        .find(|(name, _)| *name == sig)
        .map(|(_, words)| *words)
}

/// Runs the decision call through the journal (see [`journaled`]) under
/// [`STEP`] and maps its answers. Returns the summary too, for its usage
/// and the raw probabilities.
pub async fn run(
    ctx: &WorkflowContext<'_>,
    model: &str,
    issue: &Issue,
    repo: &Repo,
) -> Result<(Triage, DecisionSummary), TerminalError> {
    let summary = journaled(ctx, STEP, model, issue, repo).await?;
    Ok((to_triage(&summary, repo), summary))
}

/// The decision call with every model call in its own `ctx.run`, so a
/// retried or resumed invocation never repeats (or pays again for) a call
/// that already succeeded.
///
/// The whole request is journaled as `step`. When it doesn't fit the
/// model's context window, each of the [`split`] groups is journaled as
/// `{step}-g{i}` (see [`decide_group`]), and the answers and usage are
/// added up here, outside the journal.
pub async fn journaled(
    ctx: &WorkflowContext<'_>,
    step: &str,
    model: &str,
    issue: &Issue,
    repo: &Repo,
) -> Result<DecisionSummary, TerminalError> {
    let whole = {
        let (model, issue, repo) = (model.to_owned(), issue.clone(), repo.clone());
        async move { decide_whole(&model, &issue, &repo).await }
    };
    let whole: Option<DecisionSummary> = journal(ctx, step, whole).await?;
    if let Some(summary) = whole {
        return Ok(summary);
    }
    let mut summary = empty_summary();
    for i in 0..group_count(model, issue, repo) {
        let call = {
            let (model, issue, repo) = (model.to_owned(), issue.clone(), repo.clone());
            async move { decide_group(&model, &issue, &repo, i).await }
        };
        let part: DecisionSummary = journal(ctx, &format!("{step}-g{i}"), call).await?;
        summary.answers.extend(part.answers);
        summary.usage += &part.usage;
    }
    Ok(summary)
}

fn empty_summary() -> DecisionSummary {
    DecisionSummary {
        answers: Default::default(),
        usage: Default::default(),
    }
}

/// How many [`split`] groups the request has; the same at every body
/// budget, since only the body changes.
fn group_count(model: &str, issue: &Issue, repo: &Repo) -> usize {
    split(&request_with_body_limit(
        model,
        issue,
        repo,
        BODY_BUDGETS[0],
    ))
    .len()
}

/// Question groups for a model whose context window can't hold the issue
/// and every question at once. Each prompt the model reads is the input plus
/// all the questions of its request, so smaller requests leave more room for
/// the issue body. The area options alone are about a third of the questions.
pub const GROUPS: [&[&str]; 3] = [
    &[
        q::KIND,
        q::SEVERITY,
        q::IS_REGRESSION,
        q::SECURITY_SENSITIVE,
        q::MULTIPLE_PROBLEMS,
        q::OUTDATED_VERSION,
    ],
    &[
        q::QUALITY,
        q::HAS_VERSION,
        q::HAS_ENVIRONMENT,
        q::HAS_STEPS,
        q::HAS_EXPECTED_ACTUAL,
        q::HAS_MINIMAL_REPRO,
        q::HAS_LOGS,
        q::ENOUGH_INFO,
    ],
    &[q::AREA],
];

/// Splits a request into one request per [`GROUPS`] entry, skipping empty
/// ones. Questions in no group go with the first.
pub fn split(request: &DecisionRequest) -> Vec<DecisionRequest> {
    let group_of = |name: &str| GROUPS.iter().position(|g| g.contains(&name)).unwrap_or(0);
    (0..GROUPS.len())
        .filter_map(|i| {
            let mut part = request.clone();
            part.questions.retain(|(name, _)| group_of(name) == i);
            (!part.questions.is_empty()).then_some(part)
        })
        .collect()
}

/// The whole request with the longest body, not journaled: one model call,
/// a `decision` span. `Ok(None)` when the provider says it doesn't fit the
/// model's context window (tev1:0.8b on Ollama has 2050 tokens); such a
/// rejection comes before the model runs, so it costs nothing.
pub async fn decide_whole(
    model: &str,
    issue: &Issue,
    repo: &Repo,
) -> Result<Option<DecisionSummary>, CallError> {
    let whole = request_with_body_limit(model, issue, repo, BODY_BUDGETS[0]);
    match models::decide(STEP, whole).await {
        Err(e) if exceeds_context(&e) => Ok(None),
        other => other.map(Some),
    }
}

/// The `i`-th [`split`] group, not journaled: steps down through
/// [`BODY_BUDGETS`] until it fits. Only the attempt that fits runs the
/// model, so this is one billed call.
pub async fn decide_group(
    model: &str,
    issue: &Issue,
    repo: &Repo,
    i: usize,
) -> Result<DecisionSummary, CallError> {
    let mut result = Err(CallError::terminal("no body budget"));
    for budget in BODY_BUDGETS {
        let mut parts = split(&request_with_body_limit(model, issue, repo, budget));
        if i >= parts.len() {
            return Err(CallError::terminal(format!("no question group {i}")));
        }
        result = models::decide(STEP, parts.swap_remove(i)).await;
        match &result {
            Err(e) if exceeds_context(e) => continue,
            _ => break,
        }
    }
    result
}

/// The decision call outside any journal, for tests and probes: the whole
/// request, then each group when it doesn't fit. Inside a workflow use
/// [`journaled`], which journals every call on its own.
pub async fn decide(model: &str, issue: &Issue, repo: &Repo) -> Result<DecisionSummary, CallError> {
    if let Some(summary) = decide_whole(model, issue, repo).await? {
        return Ok(summary);
    }
    let mut summary = empty_summary();
    for i in 0..group_count(model, issue, repo) {
        let part = decide_group(model, issue, repo, i).await?;
        summary.answers.extend(part.answers);
        summary.usage += &part.usage;
    }
    Ok(summary)
}

/// allama's `Display` of `Error::ContextLengthExceeded`, which reaches here
/// only as the message of a `CallError`.
fn exceeds_context(e: &CallError) -> bool {
    e.to_string().contains("context length exceeded")
}

/// Maps the decision answers onto `Triage`.
///
/// Confidence is `max(p, 1 - p)` for a yes/no answer; for a choice or score,
/// the provider's own confidence when it reports one, else the probability
/// of the answer given. A question the model didn't answer leaves its field
/// out. Like the other strategies, the reproducibility checklist is only
/// given for bug-like kinds (bug, failing_test, flake) or an unknown kind;
/// its confidence is that of `enough_info`, the verdict that is scored.
/// `duplicates` and `suggested_reply` are always left out.
pub fn to_triage(summary: &DecisionSummary, repo: &Repo) -> Triage {
    let kind = choice(summary, q::KIND)
        .and_then(|(key, c)| Kind::parse(key).map(|k| Field::new(k).with_confidence(c)));
    let area = choice(summary, q::AREA).and_then(|(key, c)| {
        repo.labels_with_prefix("sig/")
            .any(|l| l.name == key)
            .then(|| Field::new(key.to_owned()).with_confidence(c))
    });
    let severity = choice(summary, q::SEVERITY)
        .and_then(|(key, c)| Severity::parse(key).map(|s| Field::new(s).with_confidence(c)));
    let quality = match summary.answers.get(q::QUALITY) {
        Some(
            a @ DecisionAnswer::Score {
                most_likely: Some(level),
                confidence,
                ..
            },
        ) if *level < QUALITY_RUBRIC.len() => Some(
            Field::new(QUALITY_RUBRIC[*level].0)
                .with_confidence(confidence.unwrap_or_else(|| a.top_probability())),
        ),
        _ => None,
    };
    let scope = yes_no(summary, q::MULTIPLE_PROBLEMS).map(|(multiple, c)| {
        Field::new(if multiple {
            Scope::Multiple
        } else {
            Scope::Single
        })
        .with_confidence(c)
    });
    let outdated_version = repo.latest_release.as_deref().and_then(|latest| {
        yes_no(summary, q::OUTDATED_VERSION).map(|(outdated, c)| {
            Field::new(outdated)
                .with_confidence(c)
                .with_evidence(format!("compared with latest release {latest}"))
        })
    });
    let reproducibility = bug_like_kind(kind.as_ref().map(|k| k.value))
        .then(|| reproducibility(summary))
        .flatten();
    Triage {
        kind,
        area,
        severity,
        is_regression: yes_no_field(summary, q::IS_REGRESSION),
        security_sensitive: yes_no_field(summary, q::SECURITY_SENSITIVE),
        quality,
        reproducibility,
        outdated_version,
        scope,
        duplicates: None,
        suggested_reply: None,
    }
}

/// The checklist, when the model answered every item and the verdict. The
/// evidence names the items the model was least sure about, which is what a
/// verifier should look at first.
fn reproducibility(summary: &DecisionSummary) -> Option<Field<Reproducibility>> {
    let mut checks = [false; 6];
    let mut unsure: Vec<(&str, f64)> = Vec::new();
    for (i, (name, _)) in CHECKLIST.iter().enumerate() {
        let (present, c) = yes_no(summary, name)?;
        checks[i] = present;
        if c < 0.8 {
            unsure.push((name, c));
        }
    }
    let (enough_info, c) = yes_no(summary, q::ENOUGH_INFO)?;
    let [
        version,
        environment,
        steps,
        expected_actual,
        minimal_repro,
        logs,
    ] = checks;
    let mut field = Field::new(Reproducibility::from_checks(
        version,
        environment,
        steps,
        expected_actual,
        minimal_repro,
        logs,
        enough_info,
    ))
    .with_confidence(c);
    if !unsure.is_empty() {
        unsure.sort_by(|a, b| a.1.total_cmp(&b.1));
        let mut evidence = String::from("least sure about:");
        for (name, c) in unsure {
            let _ = write!(evidence, " {name} ({c:.2})");
        }
        field = field.with_evidence(evidence);
    }
    Some(field)
}

/// The chosen option and its confidence.
fn choice<'a>(summary: &'a DecisionSummary, name: &str) -> Option<(&'a str, f64)> {
    match summary.answers.get(name)? {
        a @ DecisionAnswer::Choice {
            chosen, confidence, ..
        } => Some((chosen, confidence.unwrap_or_else(|| a.top_probability()))),
        _ => None,
    }
}

/// The answer (`p >= 0.5` is yes) and `max(p, 1 - p)`.
fn yes_no(summary: &DecisionSummary, name: &str) -> Option<(bool, f64)> {
    summary.yes_no(name).map(|p| (p >= 0.5, p.max(1.0 - p)))
}

fn yes_no_field(summary: &DecisionSummary, name: &str) -> Option<Field<bool>> {
    yes_no(summary, name).map(|(value, c)| Field::new(value).with_confidence(c))
}

/// The [`FIELDS`] whose confidence is below their threshold
/// (`thresholds.accept(field)`), in `FIELDS` order. A field that is missing,
/// or has no confidence, counts as low: nothing vouches for it.
pub fn low_confidence_fields(triage: &Triage, thresholds: &Thresholds) -> Vec<&'static str> {
    FIELDS
        .into_iter()
        .filter(|&name| match field_confidence(triage, name) {
            Some(Some(c)) => c < thresholds.accept(name),
            _ => true,
        })
        .collect()
}

/// The [`low_confidence_fields`] that a second opinion could actually fill
/// in or change, for the strategies that escalate (`Workflow`, `DraftVerify`).
/// Skipped, because the draft leaves them out on purpose rather than for
/// lack of confidence:
///
/// - `outdated_version` without `repo.latest_release`: nothing to compare
///   the reported version against.
/// - `area` when the repo has fewer than two SIG labels: there is nothing to
///   choose between.
/// - `reproducibility` when `kind` is confidently not bug-like: the checklist
///   is only given for bug-like kinds, so a confident feature request has
///   none. If `kind` is low too it is kept, since the new kind may be a bug.
pub fn escalatable_fields(
    draft: &Triage,
    thresholds: &Thresholds,
    repo: &Repo,
) -> Vec<&'static str> {
    let low = low_confidence_fields(draft, thresholds);
    let kind_settled = !low.contains(&"kind");
    let sigs = repo.labels_with_prefix("sig/").take(2).count();
    low.into_iter()
        .filter(|&field| match field {
            "outdated_version" => repo.latest_release.is_some(),
            "area" => sigs >= 2,
            "reproducibility" => !(kind_settled && !bug_like(draft)),
            _ => true,
        })
        .collect()
}

/// Bug, failing test, flake, or no kind at all: the kinds a reproducibility
/// checklist applies to.
pub fn bug_like(triage: &Triage) -> bool {
    bug_like_kind(triage.kind.as_ref().map(|k| k.value))
}

fn bug_like_kind(kind: Option<Kind>) -> bool {
    matches!(
        kind,
        Some(Kind::Bug | Kind::FailingTest | Kind::Flake) | None
    )
}

/// `None` when the field is missing, `Some(None)` when it has no confidence.
pub fn field_confidence(triage: &Triage, name: &str) -> Option<Option<f64>> {
    fn c<T>(f: &Option<Field<T>>) -> Option<Option<f64>> {
        f.as_ref().map(|f| f.confidence)
    }
    match name {
        "kind" => c(&triage.kind),
        "area" => c(&triage.area),
        "severity" => c(&triage.severity),
        "is_regression" => c(&triage.is_regression),
        "security_sensitive" => c(&triage.security_sensitive),
        "quality" => c(&triage.quality),
        "reproducibility" => c(&triage.reproducibility),
        "outdated_version" => c(&triage.outdated_version),
        "scope" => c(&triage.scope),
        "duplicates" => c(&triage.duplicates),
        "suggested_reply" => c(&triage.suggested_reply),
        _ => None,
    }
}

/// `overrides` over `base`, field by field: a field that is `Some` in
/// `overrides` replaces the one in `base`, whole (value, confidence and
/// evidence); a `None` keeps `base`'s.
pub fn merge(base: Triage, overrides: Triage) -> Triage {
    Triage {
        kind: overrides.kind.or(base.kind),
        area: overrides.area.or(base.area),
        severity: overrides.severity.or(base.severity),
        is_regression: overrides.is_regression.or(base.is_regression),
        security_sensitive: overrides.security_sensitive.or(base.security_sensitive),
        quality: overrides.quality.or(base.quality),
        reproducibility: overrides.reproducibility.or(base.reproducibility),
        outdated_version: overrides.outdated_version.or(base.outdated_version),
        scope: overrides.scope.or(base.scope),
        duplicates: overrides.duplicates.or(base.duplicates),
        suggested_reply: overrides.suggested_reply.or(base.suggested_reply),
    }
}

/// The value of a field as compact JSON, with its confidence, for prompts
/// that show a draft to a model: `"bug" (0.93)`, `missing`.
pub fn describe_field(triage: &Triage, name: &str) -> String {
    fn show<T: Serialize>(f: &Option<Field<T>>) -> String {
        match f {
            None => "missing".to_owned(),
            Some(f) => {
                let value = serde_json::to_string(&f.value).unwrap_or_default();
                match f.confidence {
                    Some(c) => format!("{value} (confidence {c:.2})"),
                    None => format!("{value} (no confidence)"),
                }
            }
        }
    }
    match name {
        "kind" => show(&triage.kind),
        "area" => show(&triage.area),
        "severity" => show(&triage.severity),
        "is_regression" => show(&triage.is_regression),
        "security_sensitive" => show(&triage.security_sensitive),
        "quality" => show(&triage.quality),
        "reproducibility" => show(&triage.reproducibility),
        "outdated_version" => show(&triage.outdated_version),
        "scope" => show(&triage.scope),
        "duplicates" => show(&triage.duplicates),
        "suggested_reply" => show(&triage.suggested_reply),
        _ => "unknown field".to_owned(),
    }
}

/// Matches a model's area answer to a SIG label: accepts `sig/node`,
/// `node`, `SIG Node`, `sig-node`; anything else is `None`.
pub fn match_sig(answer: &str, repo: &Repo) -> Option<String> {
    let wanted = answer
        .trim()
        .to_lowercase()
        .replace("sig-", "sig/")
        .replace("sig ", "sig/");
    let wanted = wanted
        .strip_prefix("sig/")
        .unwrap_or(&wanted)
        .replace(' ', "-");
    if wanted.is_empty() {
        return None;
    }
    repo.labels_with_prefix("sig/")
        .find(|l| l.name["sig/".len()..].eq_ignore_ascii_case(&wanted))
        .map(|l| l.name.clone())
}

pub(crate) fn truncate(text: &str, max_chars: usize) -> String {
    match text.char_indices().nth(max_chars) {
        Some((i, _)) => format!("{}\n[... truncated]", &text[..i]),
        None => text.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use triage_core::{Label, Usage};

    pub(crate) fn repo() -> Repo {
        let mut labels: Vec<Label> = ["sig/node", "sig/network", "sig/storage"]
            .into_iter()
            .map(|name| Label {
                name: name.into(),
                description: Some(format!("Categorizes an issue or PR as relevant to {name}.")),
            })
            .collect();
        labels.push(Label {
            name: "kind/bug".into(),
            description: None,
        });
        Repo {
            name: "kubernetes/kubernetes".into(),
            labels,
            latest_release: Some("v1.33.2".into()),
            as_of: "2025-07-01T00:00:00Z".into(),
        }
    }

    pub(crate) fn issue() -> Issue {
        Issue {
            number: 42,
            title: "kubelet OOMKilled after upgrade".into(),
            body: "After upgrading to v1.32.1 the kubelet restarts pods.".into(),
            author_association: "NONE".into(),
            created_at: "2025-07-01T00:00:00Z".into(),
        }
    }

    fn choice_answer(chosen: &str, p: f64, confidence: Option<f64>) -> DecisionAnswer {
        DecisionAnswer::Choice {
            chosen: chosen.into(),
            confidence,
            probabilities: [(chosen.to_owned(), p)].into(),
        }
    }

    fn yes(p: f64) -> DecisionAnswer {
        DecisionAnswer::YesNo { probability: p }
    }

    /// A summary as the decision model would return it for a bug report.
    pub(crate) fn summary() -> DecisionSummary {
        let mut answers: BTreeMap<String, DecisionAnswer> = BTreeMap::new();
        answers.insert(q::KIND.into(), choice_answer("bug", 0.97, Some(0.9)));
        answers.insert(q::AREA.into(), choice_answer("sig/node", 0.6, None));
        answers.insert(
            q::SEVERITY.into(),
            choice_answer("important_soon", 0.4, Some(0.3)),
        );
        answers.insert(q::IS_REGRESSION.into(), yes(0.92));
        answers.insert(q::SECURITY_SENSITIVE.into(), yes(0.03));
        answers.insert(q::MULTIPLE_PROBLEMS.into(), yes(0.3));
        answers.insert(q::OUTDATED_VERSION.into(), yes(0.85));
        answers.insert(
            q::QUALITY.into(),
            DecisionAnswer::Score {
                score: 2.6,
                most_likely: Some(3),
                confidence: None,
                probabilities: vec![0.0, 0.1, 0.3, 0.5, 0.1],
            },
        );
        for (name, p) in [
            (q::HAS_VERSION, 0.99),
            (q::HAS_ENVIRONMENT, 0.2),
            (q::HAS_STEPS, 0.7),
            (q::HAS_EXPECTED_ACTUAL, 0.9),
            (q::HAS_MINIMAL_REPRO, 0.1),
            (q::HAS_LOGS, 0.6),
            (q::ENOUGH_INFO, 0.75),
        ] {
            answers.insert(name.into(), yes(p));
        }
        DecisionSummary {
            answers,
            usage: Usage {
                decision_calls: 1,
                ..Usage::default()
            },
        }
    }

    #[test]
    fn request_answers_every_field_within_limits() {
        let r = request("ollama:tev1:0.8b-q8_0", &issue(), &repo());
        let names: Vec<&str> = r.questions.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names.len(), 15, "{names:?}");
        assert_eq!(names[0], q::KIND);
        assert!(names.contains(&q::OUTDATED_VERSION));
        let limits = allama::DecisionLimits::new()
            .max_questions(64)
            .max_choice_options(26)
            .max_score_levels(10);
        r.validate(&limits).unwrap();
        let area = match r.find_question(q::AREA).unwrap() {
            Question::Choice(c) => c,
            other => panic!("{other:?}"),
        };
        assert_eq!(area.options.len(), 3);
        let node = area.find_option("sig/node").unwrap();
        let text = node.description.as_deref().unwrap();
        assert!(text.contains("kubelet"), "{text}");
        // The state keeps its key order and the latest release.
        let json = r.input.to_json_string().unwrap();
        assert!(
            json.starts_with(r#"{"issue":{"title":"kubelet OOMKilled"#),
            "{json}"
        );
        assert!(
            json.ends_with(r#""repo":{"latest_release":"v1.33.2"}}"#),
            "{json}"
        );
    }

    #[test]
    fn split_covers_every_question_once() {
        let r = request("ollama:tev1:0.8b-q8_0", &issue(), &repo());
        let parts = split(&r);
        assert_eq!(parts.len(), 3);
        let mut names: Vec<&str> = parts
            .iter()
            .flat_map(|p| p.questions.iter().map(|(n, _)| n.as_str()))
            .collect();
        assert_eq!(parts[2].questions[0].0, q::AREA);
        assert!(parts.iter().all(|p| p.input == r.input));
        names.sort();
        let mut all: Vec<&str> = r.questions.iter().map(|(n, _)| n.as_str()).collect();
        all.sort();
        assert_eq!(names, all);
    }

    #[test]
    fn request_without_release_skips_outdated() {
        let mut repo = repo();
        repo.latest_release = None;
        let r = request("ollama:tev1:0.8b-q8_0", &issue(), &repo);
        assert!(r.find_question(q::OUTDATED_VERSION).is_none());
    }

    #[test]
    fn too_many_sigs_keep_the_relevant_ones() {
        let mut repo = repo();
        for i in 0..30 {
            repo.labels.push(Label {
                name: format!("sig/extra-{i}"),
                description: Some("Extra".into()),
            });
        }
        let options = sig_options(&issue(), &repo);
        assert_eq!(options.len(), MAX_CHOICE_OPTIONS);
        assert_eq!(options[0].0, "sig/node");
        request("ollama:tev1:0.8b-q8_0", &issue(), &repo)
            .validate(&allama::DecisionLimits::new().max_choice_options(26))
            .unwrap();
    }

    #[test]
    fn maps_answers_with_confidences() {
        let t = to_triage(&summary(), &repo());
        let kind = t.kind.unwrap();
        assert_eq!((kind.value, kind.confidence), (Kind::Bug, Some(0.9)));
        // No provider confidence: the top probability.
        let area = t.area.unwrap();
        assert_eq!(
            (area.value.as_str(), area.confidence),
            ("sig/node", Some(0.6))
        );
        assert_eq!(t.severity.unwrap().value, Severity::ImportantSoon);
        let regression = t.is_regression.unwrap();
        assert!(regression.value);
        assert_eq!(regression.confidence, Some(0.92));
        let security = t.security_sensitive.unwrap();
        assert!(!security.value);
        assert_eq!(security.confidence, Some(0.97));
        let quality = t.quality.unwrap();
        assert_eq!((quality.value, quality.confidence), (4, Some(0.5)));
        let scope = t.scope.unwrap();
        assert_eq!(scope.value, Scope::Single);
        assert!((scope.confidence.unwrap() - 0.7).abs() < 1e-9);
        assert!(t.outdated_version.unwrap().value);
        let repro = t.reproducibility.unwrap();
        assert!(repro.value.enough_info && repro.value.has_version && !repro.value.has_environment);
        assert_eq!(repro.value.missing.len(), 2);
        assert_eq!(repro.confidence, Some(0.75));
        let evidence = repro.evidence.unwrap();
        assert!(
            evidence.starts_with("least sure about: has_logs (0.60)"),
            "{evidence}"
        );
        assert!(t.duplicates.is_none() && t.suggested_reply.is_none());
    }

    #[test]
    fn invalid_or_missing_answers_are_left_out() {
        let mut s = summary();
        s.answers
            .insert(q::AREA.into(), choice_answer("sig/windows", 0.9, None));
        s.answers
            .insert(q::KIND.into(), choice_answer("feature", 0.9, None));
        s.answers.remove(q::HAS_LOGS);
        s.answers.remove(q::SEVERITY);
        let t = to_triage(&s, &repo());
        assert!(t.area.is_none() && t.severity.is_none());
        assert_eq!(t.kind.unwrap().value, Kind::Feature);
        assert!(t.reproducibility.is_none());

        // Reproducibility is only given for bug-like kinds.
        let mut s = summary();
        s.answers
            .insert(q::KIND.into(), choice_answer("feature", 0.9, None));
        assert!(to_triage(&s, &repo()).reproducibility.is_none());
    }

    #[test]
    fn low_confidence_uses_per_field_thresholds() {
        let t = to_triage(&summary(), &repo());
        let mut thresholds = Thresholds::default();
        assert_eq!(
            low_confidence_fields(&t, &thresholds),
            vec!["area", "severity", "quality", "reproducibility", "scope"]
        );
        thresholds.per_field.insert("area".into(), 0.5);
        thresholds.per_field.insert("kind".into(), 0.95);
        assert_eq!(
            low_confidence_fields(&t, &thresholds),
            vec!["kind", "severity", "quality", "reproducibility", "scope"]
        );

        // Missing fields and fields without a confidence count as low.
        let mut t = t;
        t.area = None;
        t.kind.as_mut().unwrap().confidence = None;
        thresholds.accept = 0.0;
        thresholds.per_field.clear();
        assert_eq!(low_confidence_fields(&t, &thresholds), vec!["kind", "area"]);
    }

    /// A confident draft: every answer at 0.99.
    fn confident(kind: &str) -> DecisionSummary {
        let mut s = summary();
        for answer in s.answers.values_mut() {
            *answer = match answer {
                DecisionAnswer::Choice { chosen, .. } => choice_answer(chosen, 0.99, Some(0.99)),
                DecisionAnswer::YesNo { probability } if *probability >= 0.5 => yes(0.99),
                DecisionAnswer::YesNo { .. } => yes(0.01),
                DecisionAnswer::Score { .. } => DecisionAnswer::Score {
                    score: 3.0,
                    most_likely: Some(3),
                    confidence: Some(0.99),
                    probabilities: vec![0.0, 0.0, 0.0, 0.99, 0.01],
                },
            };
        }
        s.answers
            .insert(q::KIND.into(), choice_answer(kind, 0.99, Some(0.99)));
        s
    }

    #[test]
    fn escalation_skips_fields_left_out_on_purpose() {
        let thresholds = Thresholds::default();
        let repo = repo();

        // A confident feature request has no checklist, which counts as low
        // but can't be filled in: nothing is left to verify.
        let feature = to_triage(&confident("feature"), &repo);
        assert!(feature.reproducibility.is_none());
        assert_eq!(
            low_confidence_fields(&feature, &thresholds),
            vec!["reproducibility"]
        );
        assert!(escalatable_fields(&feature, &thresholds, &repo).is_empty());

        // Unless `kind` is unsure too: the verified kind may be a bug.
        let mut unsure = feature.clone();
        unsure.kind.as_mut().unwrap().confidence = Some(0.5);
        assert_eq!(
            escalatable_fields(&unsure, &thresholds, &repo),
            vec!["kind", "reproducibility"]
        );

        // A confident bug keeps an unsure checklist.
        let mut bug = to_triage(&confident("bug"), &repo);
        assert!(escalatable_fields(&bug, &thresholds, &repo).is_empty());
        bug.reproducibility.as_mut().unwrap().confidence = Some(0.5);
        assert_eq!(
            escalatable_fields(&bug, &thresholds, &repo),
            vec!["reproducibility"]
        );

        // No latest release: `outdated_version` is never asked.
        let mut no_release = repo.clone();
        no_release.latest_release = None;
        let t = to_triage(&confident("feature"), &no_release);
        assert_eq!(
            low_confidence_fields(&t, &thresholds),
            vec!["reproducibility", "outdated_version"]
        );
        assert!(escalatable_fields(&t, &thresholds, &no_release).is_empty());

        // Fewer than two SIGs: `area` is never asked.
        let mut one_sig = repo.clone();
        one_sig
            .labels
            .retain(|l| l.name == "sig/node" || l.name == "kind/bug");
        let mut t = to_triage(&confident("bug"), &one_sig);
        t.area = None;
        assert_eq!(low_confidence_fields(&t, &thresholds), vec!["area"]);
        assert!(escalatable_fields(&t, &thresholds, &one_sig).is_empty());
        assert_eq!(escalatable_fields(&t, &thresholds, &repo), vec!["area"]);
    }

    #[test]
    fn merge_prefers_overrides() {
        let base = to_triage(&summary(), &repo());
        let overrides = Triage {
            area: Some(Field::new("sig/network".to_owned()).with_evidence("verified")),
            duplicates: Some(Field::new(vec![7])),
            ..Triage::default()
        };
        let merged = merge(base, overrides);
        assert_eq!(merged.area.unwrap().value, "sig/network");
        assert_eq!(merged.duplicates.unwrap().value, vec![7]);
        assert_eq!(merged.kind.unwrap().value, Kind::Bug);
        assert!(merged.suggested_reply.is_none());
    }

    #[test]
    fn describes_fields_for_prompts() {
        let t = to_triage(&summary(), &repo());
        assert_eq!(describe_field(&t, "kind"), r#""bug" (confidence 0.90)"#);
        assert_eq!(describe_field(&t, "duplicates"), "missing");
    }

    #[test]
    fn matches_sig_answers() {
        let repo = repo();
        for answer in ["sig/node", "node", "SIG Node", "sig-node", " Sig/Node "] {
            assert_eq!(
                match_sig(answer, &repo).as_deref(),
                Some("sig/node"),
                "{answer}"
            );
        }
        assert!(match_sig("sig/windows", &repo).is_none());
        assert!(match_sig("", &repo).is_none());
    }
}
