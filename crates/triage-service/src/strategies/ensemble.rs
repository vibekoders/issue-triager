//! `Ensemble`: several voters (single-shot LLM calls and/or decision-model
//! calls), aggregated per field by a confidence-weighted vote in code or by an
//! LLM judge that reads the issue and all votes.
//!
//! Config: `voters[]` (`{kind: llm, model, temperature?}` or `{kind:
//! decision, model}`), `aggregator` (`vote` or `{kind: judge, model}`).
//! Compares 3 voters against 1, and vote against judge: buying accuracy with
//! more calls.
//!
//! Each voter is journaled as `voter-<i>`: an LLM voter is exactly
//! SingleShot with the voter's model and temperature, one call; a decision
//! voter is the decision triage, whose question groups (when the model's
//! context is short) are journaled on their own as `voter-<i>-g<j>`. The
//! voters run one after another: restate-sdk 0.12 can't replay concurrent
//! runs (see the `workflow` module). A voter that fails is left out; the
//! triage fails only when every voter does.
//!
//! The vote, per field: each voter's answer weighs its confidence (an LLM's
//! self-reported one, or 0.5 without one; a decision model's calibrated
//! one). The heaviest answer wins, and its confidence is its share of the
//! total weight, so agreement shows up as confidence. Quality is the
//! weighted median, the reproducibility checklist is voted item by item, and
//! the reply comes from an LLM voter that agrees with the winning kind. The
//! evidence records the tally, e.g. `votes: bug 2 (1.85), feature 1 (0.55)`.
//!
//! The judge sees SingleShot's prompt plus every voter's answers, and fills
//! the same schema; each field's evidence keeps the vote tally next to the
//! judge's answer. If the judge's call fails, the vote stands.

use std::fmt::Write;

use restate_sdk::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::{Instrument, info, warn};
use triage_core::config::{Aggregator, EnsembleConfig, SingleShotConfig, Voter};
use triage_core::{
    Field, Issue, Kind, Repo, Reproducibility, Triage, TriageRequest, TriageResponse, Usage,
};

use super::agent::check_address;
use super::decision_triage;
use super::single_shot::{self, ModelTriage};
use super::{parse_config, record_run, response};
use crate::models::{self, ChatRequest, Structured, journal};
use crate::trace::handler_span;

pub const VERSION: &str = "3";
const NAME: &str = "Ensemble";
const JUDGE_STEP: &str = "judge";

/// The weight of an answer that comes without a confidence.
const DEFAULT_WEIGHT: f64 = 0.5;
/// The least weight an answer gets, so a voter claiming zero confidence
/// still counts as a vote and the total is never zero.
const MIN_WEIGHT: f64 = 0.01;
/// A ceiling with room for a reasoning model's reasoning tokens.
const JUDGE_OUTPUT_TOKENS: u32 = 6000;

pub struct EnsembleStrategy;

/// Timeouts: see "Restate timeouts" in the [`super`] module docs.
#[restate_sdk::workflow(name = "Ensemble", inactivity_timeout = "5m", abort_timeout = "1h")]
impl EnsembleStrategy {
    #[handler]
    async fn run(
        &self,
        ctx: WorkflowContext<'_>,
        Json(request): Json<TriageRequest>,
    ) -> Result<Json<TriageResponse>, HandlerError> {
        let span = handler_span(NAME, ctx.key(), ctx.headers());
        async move {
            let (request, config_hash) = parse_config::<EnsembleConfig>(request)?;
            record_run(&tracing::Span::current(), VERSION, &config_hash);
            let config = &request.config;
            if config.voters.is_empty() {
                return Err(TerminalError::new_with_code(
                    400,
                    "invalid config: at least one voter is needed",
                )
                .into());
            }
            for voter in &config.voters {
                check_address(voter_model(voter))?;
            }
            if let Aggregator::Judge { model } = &config.aggregator {
                check_address(model)?;
            }
            let (issue, repo) = (&request.issue, &request.repo);

            let mut usage = Usage::default();
            let mut ballots = Vec::new();
            for (i, voter) in config.voters.iter().enumerate() {
                let result = vote(&ctx, i, voter, issue, repo).await;
                match result {
                    Ok(ballot) => {
                        usage += &ballot.usage;
                        ballots.push(ballot);
                    }
                    // The failure is journaled, so every replay leaves the
                    // same voter out.
                    Err(e) => warn!(voter = i, error = %e, "voter failed; leaving it out"),
                }
            }
            if ballots.is_empty() {
                return Err(TerminalError::new("every voter failed").into());
            }

            let voted = aggregate(&ballots);
            let triage = match &config.aggregator {
                Aggregator::Vote => voted,
                Aggregator::Judge { model } => {
                    match judge(&ctx, model, issue, repo, &ballots).await {
                        Ok(out) => {
                            usage += &out.usage;
                            annotate_judged(single_shot::to_triage(out.value, repo), &voted, model)
                        }
                        Err(e) => {
                            warn!(error = %e, "judge failed; using the weighted vote");
                            voted
                        }
                    }
                }
            };
            info!(
                voters = ballots.len() as i64,
                kind = triage.kind.as_ref().map(|k| k.value.as_str()),
                kind_votes = triage.kind.as_ref().and_then(|k| k.evidence.as_deref()),
                cost_usd = usage.cost_usd,
                "ensemble triage done"
            );
            Ok(Json(response(NAME, VERSION, config_hash, triage, usage)))
        }
        .instrument(span)
        .await
    }
}

fn voter_model(voter: &Voter) -> &str {
    match voter {
        Voter::Llm { model, .. } | Voter::Decision { model } => model,
    }
}

/// One voter's answer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ballot {
    /// Who voted, for evidence and the judge: `llm ollama:qwen3:1.7b
    /// (temperature 0.2)`.
    pub voter: String,
    pub triage: Triage,
    pub usage: Usage,
}

fn voter_label(voter: &Voter) -> String {
    match voter {
        Voter::Llm {
            model,
            temperature: Some(t),
        } => format!("llm {model} (temperature {t})"),
        Voter::Llm { model, .. } => format!("llm {model}"),
        Voter::Decision { model } => format!("decision {model}"),
    }
}

/// One voter's call, journaled as `voter-<i>` (a decision voter's groups as
/// `voter-<i>-g<j>`).
async fn vote(
    ctx: &WorkflowContext<'_>,
    index: usize,
    voter: &Voter,
    issue: &Issue,
    repo: &Repo,
) -> Result<Ballot, TerminalError> {
    let step = format!("voter-{index}");
    let label = voter_label(voter);
    match voter {
        Voter::Llm { model, temperature } => {
            let request = single_shot::chat_request(&TriageRequest {
                config: SingleShotConfig {
                    model: model.clone(),
                    temperature: *temperature,
                },
                issue: issue.clone(),
                repo: repo.clone(),
            });
            let out: Structured<ModelTriage> =
                journal(ctx, &step, models::extract(&step, request)).await?;
            Ok(Ballot {
                voter: label,
                triage: single_shot::to_triage(out.value, repo),
                usage: out.usage,
            })
        }
        Voter::Decision { model } => {
            let summary = decision_triage::journaled(ctx, &step, model, issue, repo).await?;
            Ok(Ballot {
                voter: label,
                triage: decision_triage::to_triage(&summary, repo),
                usage: summary.usage,
            })
        }
    }
}

fn weight<T>(field: &Field<T>) -> f64 {
    field
        .confidence
        .filter(|c| c.is_finite())
        .unwrap_or(DEFAULT_WEIGHT)
        .clamp(MIN_WEIGHT, 1.0)
}

/// The confidence-weighted vote over every field.
pub fn aggregate(ballots: &[Ballot]) -> Triage {
    let triages: Vec<&Triage> = ballots.iter().map(|b| &b.triage).collect();
    fn votes<'a, T: Clone + 'a>(
        triages: &[&'a Triage],
        get: impl Fn(&'a Triage) -> &'a Option<Field<T>>,
    ) -> Vec<(T, f64)> {
        triages
            .iter()
            .filter_map(|t| get(t).as_ref().map(|f| (f.value.clone(), weight(f))))
            .collect()
    }
    let kind = tally(votes(&triages, |t| &t.kind));
    let mut triage = Triage {
        area: tally(votes(&triages, |t| &t.area)),
        severity: tally(votes(&triages, |t| &t.severity)),
        is_regression: tally(votes(&triages, |t| &t.is_regression)),
        security_sensitive: tally(votes(&triages, |t| &t.security_sensitive)),
        quality: weighted_median(votes(&triages, |t| &t.quality)),
        reproducibility: checklist_vote(&votes(&triages, |t| &t.reproducibility)),
        outdated_version: tally(votes(&triages, |t| &t.outdated_version)),
        scope: tally(votes(&triages, |t| &t.scope)),
        duplicates: None,
        suggested_reply: pick_reply(ballots, kind.as_ref().map(|k| k.value)),
        kind,
    };
    // Like the other strategies: no checklist unless the issue is bug-like.
    if !matches!(
        triage.kind.as_ref().map(|k| k.value),
        Some(Kind::Bug | Kind::FailingTest | Kind::Flake) | None
    ) {
        triage.reproducibility = None;
    }
    triage
}

/// The weighted plurality. Ties go to the answer seen first, i.e. the
/// earlier voter in the config.
fn tally<T: PartialEq + Serialize>(votes: Vec<(T, f64)>) -> Option<Field<T>> {
    let total: f64 = votes.iter().map(|(_, w)| w).sum();
    let mut groups: Vec<(T, f64, usize)> = Vec::new();
    for (value, w) in votes {
        match groups.iter_mut().find(|(v, _, _)| *v == value) {
            Some(group) => {
                group.1 += w;
                group.2 += 1;
            }
            None => groups.push((value, w, 1)),
        }
    }
    let evidence = votes_evidence(&groups);
    let mut winner: Option<(T, f64)> = None;
    for (value, w, _) in groups {
        if winner.as_ref().is_none_or(|(_, best)| w > *best) {
            winner = Some((value, w));
        }
    }
    let (value, w) = winner?;
    Some(
        Field::new(value)
            .with_confidence(w / total)
            .with_evidence(evidence),
    )
}

/// `votes: bug 2 (1.85), feature 1 (0.55)`: each answer with its count and
/// summed weight.
fn votes_evidence<T: Serialize>(groups: &[(T, f64, usize)]) -> String {
    let parts: Vec<String> = groups
        .iter()
        .map(|(value, w, n)| format!("{} {n} ({w:.2})", display(value)))
        .collect();
    format!("votes: {}", parts.join(", "))
}

fn display<T: Serialize>(value: &T) -> String {
    serde_json::to_string(value)
        .unwrap_or_default()
        .trim_matches('"')
        .to_owned()
}

/// The lowest score at which half the weight is reached. Its confidence is
/// the share of the weight that voted exactly that score.
fn weighted_median(mut votes: Vec<(u8, f64)>) -> Option<Field<u8>> {
    if votes.is_empty() {
        return None;
    }
    let total: f64 = votes.iter().map(|(_, w)| w).sum();
    let mut groups: Vec<(u8, f64, usize)> = Vec::new();
    for &(value, w) in &votes {
        match groups.iter_mut().find(|(v, _, _)| *v == value) {
            Some(group) => {
                group.1 += w;
                group.2 += 1;
            }
            None => groups.push((value, w, 1)),
        }
    }
    votes.sort_by_key(|(value, _)| *value);
    let mut seen = 0.0;
    let median = votes
        .iter()
        .find(|(_, w)| {
            seen += w;
            seen >= total / 2.0 - 1e-9
        })
        .map(|(value, _)| *value)?;
    let share = groups
        .iter()
        .find(|(v, _, _)| *v == median)
        .map_or(0.0, |g| g.1)
        / total;
    Some(
        Field::new(median)
            .with_confidence(share)
            .with_evidence(format!("{} (weighted median)", votes_evidence(&groups))),
    )
}

/// The checklist voted item by item, each voter weighing its checklist's
/// confidence. The confidence is the `enough_info` verdict's share, the
/// item that is scored.
fn checklist_vote(votes: &[(Reproducibility, f64)]) -> Option<Field<Reproducibility>> {
    if votes.is_empty() {
        return None;
    }
    let item = |get: fn(&Reproducibility) -> bool| {
        tally(votes.iter().map(|(r, w)| (get(r), *w)).collect())
            .expect("there is at least one vote")
    };
    let items = [
        ("has_version", item(|r| r.has_version)),
        ("has_environment", item(|r| r.has_environment)),
        ("has_steps", item(|r| r.has_steps)),
        ("has_expected_actual", item(|r| r.has_expected_actual)),
        ("has_minimal_repro", item(|r| r.has_minimal_repro)),
        ("has_logs", item(|r| r.has_logs)),
    ];
    let enough = item(|r| r.enough_info);
    let split: Vec<&str> = items
        .iter()
        .filter(|(_, f)| f.confidence.is_some_and(|c| c < 1.0))
        .map(|(name, _)| *name)
        .collect();
    let [
        version,
        environment,
        steps,
        expected_actual,
        minimal_repro,
        logs,
    ] = items.map(|(_, f)| f.value);
    let mut evidence = format!(
        "enough_info {}",
        enough
            .evidence
            .as_deref()
            .unwrap_or_default()
            .trim_start_matches("votes: ")
    );
    if !split.is_empty() {
        let _ = write!(evidence, "; split on {}", split.join(", "));
    }
    Some(
        Field::new(Reproducibility::from_checks(
            version,
            environment,
            steps,
            expected_actual,
            minimal_repro,
            logs,
            enough.value,
        ))
        .with_confidence(enough.confidence.unwrap_or(0.0))
        .with_evidence(format!("votes: {evidence}")),
    )
}

/// The reply of the LLM voter most sure of the winning kind; any voter's
/// reply when none agrees.
fn pick_reply(ballots: &[Ballot], kind: Option<Kind>) -> Option<Field<String>> {
    let with_reply = || {
        ballots
            .iter()
            .enumerate()
            .filter_map(|(i, b)| b.triage.suggested_reply.as_ref().map(|r| (i, b, r)))
    };
    let agreeing = with_reply()
        .filter(|(_, b, _)| b.triage.kind.as_ref().map(|k| k.value) == kind)
        .fold(
            None,
            |best: Option<(usize, &Ballot, &Field<String>, f64)>, (i, b, r)| {
                let w = b.triage.kind.as_ref().map_or(0.0, weight);
                match best {
                    Some(best) if best.3 >= w => Some(best),
                    _ => Some((i, b, r, w)),
                }
            },
        )
        .map(|(i, b, r, _)| (i, b, r));
    let (i, ballot, reply) = agreeing.or_else(|| with_reply().next())?;
    Some(
        Field::new(reply.value.clone()).with_evidence(format!("from voter {i} ({})", ballot.voter)),
    )
}

const JUDGE_SYSTEM_PROMPT: &str = "You are an experienced maintainer making the final triage \
of a newly filed GitHub issue. Several triagers have already voted; you see the issue and their \
answers. Fill in every field of the JSON answer, following the field guide. Where they agree and \
the issue supports it, keep their answer; where they disagree or are unsure, decide from the \
issue yourself. Do not invent facts. Confidence values are numbers from 0 to 1 saying how likely \
your answer is to be right.";

/// The judge's call, journaled as `judge`: SingleShot's prompt for the
/// judge's model, then every ballot on one line.
async fn judge(
    ctx: &WorkflowContext<'_>,
    model: &str,
    issue: &Issue,
    repo: &Repo,
    ballots: &[Ballot],
) -> Result<Structured<ModelTriage>, TerminalError> {
    let request = ChatRequest::new(model)
        .system(JUDGE_SYSTEM_PROMPT)
        .user(judge_prompt(model, issue, repo, ballots))
        .max_output_tokens(JUDGE_OUTPUT_TOKENS);
    journal(ctx, JUDGE_STEP, models::extract(JUDGE_STEP, request)).await
}

fn judge_prompt(model: &str, issue: &Issue, repo: &Repo, ballots: &[Ballot]) -> String {
    let single_shot = single_shot::chat_request(&TriageRequest {
        config: SingleShotConfig {
            model: model.to_owned(),
            temperature: None,
        },
        issue: issue.clone(),
        repo: repo.clone(),
    });
    let mut p = single_shot
        .messages
        .iter()
        .find_map(|m| m.text())
        .unwrap_or_default();
    p.push_str("\n## Votes\n\n");
    for (i, ballot) in ballots.iter().enumerate() {
        let _ = writeln!(
            p,
            "- voter {i} ({}): {}",
            ballot.voter,
            describe(&ballot.triage)
        );
    }
    p.push_str(
        "\nGive your final answer for every field. Write suggested_reply yourself, for the kind \
         you decide on.\n",
    );
    p
}

/// A ballot on one line: `kind=bug (0.90), area=sig/node (0.80), ...`.
fn describe(t: &Triage) -> String {
    fn show<T: Serialize>(name: &str, f: &Option<Field<T>>) -> Option<String> {
        f.as_ref().map(|f| match f.confidence {
            Some(c) => format!("{name}={} ({c:.2})", display(&f.value)),
            None => format!("{name}={}", display(&f.value)),
        })
    }
    let checklist = t.reproducibility.as_ref().map(|r| {
        let missing: Vec<String> = r.value.missing.iter().map(display).collect();
        format!(
            "enough_info={}, missing=[{}]",
            r.value.enough_info,
            missing.join(", ")
        )
    });
    [
        show("kind", &t.kind),
        show("area", &t.area),
        show("severity", &t.severity),
        show("is_regression", &t.is_regression),
        show("security_sensitive", &t.security_sensitive),
        show("quality", &t.quality),
        checklist,
        show("outdated_version", &t.outdated_version),
        show("scope", &t.scope),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>()
    .join(", ")
}

/// Puts the vote tally next to each of the judge's answers, so the
/// agreement level is recorded whichever aggregator ran.
fn annotate_judged(mut judged: Triage, voted: &Triage, model: &str) -> Triage {
    fn note<T>(judged: &mut Option<Field<T>>, voted: &Option<Field<T>>, model: &str) {
        let Some(field) = judged.as_mut() else {
            return;
        };
        let mut evidence = format!("judge {model}");
        if let Some(votes) = voted.as_ref().and_then(|v| v.evidence.as_deref()) {
            let _ = write!(evidence, "; {votes}");
        }
        if let Some(own) = field.evidence.as_deref().filter(|e| !e.is_empty()) {
            let _ = write!(evidence, "; {own}");
        }
        field.evidence = Some(evidence);
    }
    note(&mut judged.kind, &voted.kind, model);
    note(&mut judged.area, &voted.area, model);
    note(&mut judged.severity, &voted.severity, model);
    note(&mut judged.is_regression, &voted.is_regression, model);
    note(
        &mut judged.security_sensitive,
        &voted.security_sensitive,
        model,
    );
    note(&mut judged.quality, &voted.quality, model);
    note(&mut judged.reproducibility, &voted.reproducibility, model);
    note(&mut judged.outdated_version, &voted.outdated_version, model);
    note(&mut judged.scope, &voted.scope, model);
    note(&mut judged.suggested_reply, &None::<Field<String>>, model);
    judged
}

#[cfg(test)]
mod tests {
    use super::*;
    use triage_core::{Label, Scope, Severity};

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
            title: "kubelet panics on startup".into(),
            body: "Since v1.32.1 the kubelet panics.".into(),
            author_association: "NONE".into(),
            created_at: "2025-07-01T00:00:00Z".into(),
        }
    }

    fn checklist(logs: bool, enough_info: bool) -> Reproducibility {
        Reproducibility::from_checks(true, false, true, true, false, logs, enough_info)
    }

    fn ballot(voter: &str, kind: Kind, c: Option<f64>, quality: u8, reply: Option<&str>) -> Ballot {
        let field = |f: Field<Kind>| match c {
            Some(c) => f.with_confidence(c),
            None => f,
        };
        Ballot {
            voter: voter.into(),
            triage: Triage {
                kind: Some(field(Field::new(kind))),
                area: Some(Field::new("sig/node".to_owned()).with_confidence(0.8)),
                severity: Some(Field::new(Severity::ImportantSoon).with_confidence(0.6)),
                is_regression: Some(Field::new(true).with_confidence(0.9)),
                security_sensitive: Some(Field::new(false).with_confidence(0.9)),
                quality: Some(Field::new(quality).with_confidence(0.7)),
                reproducibility: Some(Field::new(checklist(true, true)).with_confidence(0.7)),
                outdated_version: None,
                scope: Some(Field::new(Scope::Single).with_confidence(0.9)),
                duplicates: None,
                suggested_reply: reply.map(|r| Field::new(r.to_owned())),
            },
            usage: Usage::default(),
        }
    }

    #[test]
    fn weighted_majority_with_share_as_confidence() {
        let f = tally(vec![("bug", 0.9), ("feature", 0.6), ("bug", 0.5)]).unwrap();
        assert_eq!(f.value, "bug");
        assert!((f.confidence.unwrap() - 1.4 / 2.0).abs() < 1e-9);
        assert_eq!(
            f.evidence.as_deref(),
            Some("votes: bug 2 (1.40), feature 1 (0.60)")
        );

        // One sure voter outweighs two unsure ones.
        let f = tally(vec![("bug", 0.2), ("feature", 0.95), ("bug", 0.3)]).unwrap();
        assert_eq!(f.value, "feature");

        // Ties go to the earlier voter.
        assert!(!tally(vec![(false, 0.5), (true, 0.5)]).unwrap().value);
        assert!(tally(Vec::<(bool, f64)>::new()).is_none());
    }

    #[test]
    fn weights_default_and_floor() {
        assert_eq!(weight(&Field::new(1)), DEFAULT_WEIGHT);
        assert_eq!(weight(&Field::new(1).with_confidence(0.0)), MIN_WEIGHT);
        assert_eq!(
            weight(&Field::new(1).with_confidence(f64::NAN)),
            DEFAULT_WEIGHT
        );
    }

    #[test]
    fn quality_is_the_weighted_median() {
        let f = weighted_median(vec![(2, 0.5), (5, 0.5), (4, 0.9)]).unwrap();
        assert_eq!(f.value, 4);
        assert!((f.confidence.unwrap() - 0.9 / 1.9).abs() < 1e-9);
        assert!(f.evidence.unwrap().ends_with("(weighted median)"));
        // An outlier can't drag the median: 1 is outvoted by the 4s.
        assert_eq!(
            weighted_median(vec![(4, 0.6), (1, 1.0), (4, 0.6)])
                .unwrap()
                .value,
            4
        );
        assert!(weighted_median(vec![]).is_none());
    }

    #[test]
    fn checklist_is_voted_per_item() {
        let votes = vec![
            (checklist(true, true), 0.6),
            (checklist(false, false), 0.9),
            (checklist(false, true), 0.5),
        ];
        let f = checklist_vote(&votes).unwrap();
        assert!(!f.value.has_logs);
        assert!(f.value.enough_info);
        assert!(f.value.has_version && !f.value.has_environment);
        assert!((f.confidence.unwrap() - 1.1 / 2.0).abs() < 1e-9);
        assert_eq!(
            f.evidence.as_deref(),
            Some("votes: enough_info true 2 (1.10), false 1 (0.90); split on has_logs")
        );
    }

    #[test]
    fn aggregates_every_field() {
        let ballots = vec![
            ballot(
                "llm a",
                Kind::Bug,
                Some(0.9),
                3,
                Some("Thanks, please add logs."),
            ),
            ballot(
                "llm b",
                Kind::Feature,
                None,
                4,
                Some("Thanks for the idea!"),
            ),
            ballot("decision d", Kind::Bug, Some(0.7), 5, None),
        ];
        let t = aggregate(&ballots);
        let kind = t.kind.unwrap();
        assert_eq!(kind.value, Kind::Bug);
        assert!((kind.confidence.unwrap() - 1.6 / 2.1).abs() < 1e-9);
        assert_eq!(
            kind.evidence.as_deref(),
            Some("votes: bug 2 (1.60), feature 1 (0.50)")
        );
        assert_eq!(t.area.unwrap().confidence, Some(1.0));
        assert_eq!(t.quality.unwrap().value, 4);
        assert!(t.reproducibility.is_some());
        assert!(t.outdated_version.is_none() && t.duplicates.is_none());
        let reply = t.suggested_reply.unwrap();
        assert_eq!(reply.value, "Thanks, please add logs.");
        assert_eq!(reply.evidence.as_deref(), Some("from voter 0 (llm a)"));

        // A non-bug winner drops the checklist.
        let ballots = vec![
            ballot(
                "llm b",
                Kind::Feature,
                Some(0.9),
                4,
                Some("Thanks for the idea!"),
            ),
            ballot("decision d", Kind::Bug, Some(0.6), 4, None),
        ];
        let t = aggregate(&ballots);
        assert_eq!(t.kind.unwrap().value, Kind::Feature);
        assert!(t.reproducibility.is_none());
        assert_eq!(t.suggested_reply.unwrap().value, "Thanks for the idea!");
    }

    #[test]
    fn reply_falls_back_to_any_voter() {
        let ballots = vec![
            ballot("llm b", Kind::Feature, Some(0.3), 4, Some("Thanks!")),
            ballot("decision d", Kind::Bug, Some(0.9), 4, None),
        ];
        let reply = pick_reply(&ballots, Some(Kind::Bug)).unwrap();
        assert_eq!(reply.value, "Thanks!");
        assert!(pick_reply(&ballots[1..], Some(Kind::Bug)).is_none());
    }

    #[test]
    fn judge_sees_the_votes() {
        let ballots = vec![
            ballot("llm a", Kind::Bug, Some(0.9), 3, None),
            ballot("decision d", Kind::Feature, Some(0.7), 5, None),
        ];
        let p = judge_prompt("ollama:qwen3:4b", &issue(), &repo(), &ballots);
        assert!(p.contains("## Field guide"), "{p}");
        assert!(p.contains(
            "- voter 0 (llm a): kind=bug (0.90), area=sig/node (0.80), severity=important_soon \
             (0.60), is_regression=true (0.90)"
        ));
        assert!(p.contains("enough_info=true, missing=[environment, minimal_repro]"));
        assert!(p.contains("- voter 1 (decision d): kind=feature (0.70)"));

        let voted = aggregate(&ballots);
        let judged = Triage {
            kind: Some(
                Field::new(Kind::Bug)
                    .with_confidence(0.8)
                    .with_evidence("panics"),
            ),
            ..Triage::default()
        };
        let t = annotate_judged(judged, &voted, "ollama:qwen3:4b");
        assert_eq!(
            t.kind.unwrap().evidence.as_deref(),
            Some("judge ollama:qwen3:4b; votes: bug 1 (0.90), feature 1 (0.70); panics")
        );
    }

    #[test]
    fn voter_labels() {
        assert_eq!(
            voter_label(&Voter::Llm {
                model: "ollama:qwen3:1.7b".into(),
                temperature: Some(0.2)
            }),
            "llm ollama:qwen3:1.7b (temperature 0.2)"
        );
        assert_eq!(
            voter_label(&Voter::Decision {
                model: "ollama:tev1:0.8b-q8_0".into()
            }),
            "decision ollama:tev1:0.8b-q8_0"
        );
    }
}
