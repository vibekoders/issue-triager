use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::Usage;

/// The output of every strategy.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct TriageResponse {
    pub triage: Triage,
    pub usage: Usage,
    /// The strategy's service name, e.g. `Heuristic`.
    pub strategy: String,
    /// The strategy's code version. Bumped when its logic or prompts change.
    pub version: String,
    /// The hash of the config that produced this response.
    pub config_hash: String,
}

/// The triage details. A field a strategy can't produce is `None`, never a
/// guess, so evaluators can tell "wrong" from "not attempted".
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct Triage {
    #[serde(default)]
    pub kind: Option<Field<Kind>>,
    /// One of the repo's `sig/*` labels, e.g. `sig/node`.
    #[serde(default)]
    pub area: Option<Field<String>>,
    #[serde(default)]
    pub severity: Option<Field<Severity>>,
    #[serde(default)]
    pub is_regression: Option<Field<bool>>,
    #[serde(default)]
    pub security_sensitive: Option<Field<bool>>,
    /// 1 (unusable) to 5 (exemplary), see `QUALITY_RUBRIC`.
    #[serde(default)]
    pub quality: Option<Field<u8>>,
    #[serde(default)]
    pub reproducibility: Option<Field<Reproducibility>>,
    /// Whether the reported version is older than the repo's latest release.
    #[serde(default)]
    pub outdated_version: Option<Field<bool>>,
    #[serde(default)]
    pub scope: Option<Field<Scope>>,
    /// Candidate duplicate issue numbers, most likely first.
    #[serde(default)]
    pub duplicates: Option<Field<Vec<u64>>>,
    #[serde(default)]
    pub suggested_reply: Option<Field<String>>,
}

/// The rubric for [`Triage::quality`], shared by the prompts and the judge.
pub const QUALITY_RUBRIC: [(u8, &str); 5] = [
    (
        1,
        "Unusable: no clear problem or request; spam, gibberish, or a one-liner with no context.",
    ),
    (
        2,
        "Poor: the topic is clear but key information is missing; a maintainer would have to ask several questions.",
    ),
    (
        3,
        "Adequate: understandable and mostly complete, but needs at least one follow-up question.",
    ),
    (
        4,
        "Good: clear, complete and focused; a maintainer can act on it without follow-up.",
    ),
    (
        5,
        "Exemplary: clear, complete, focused, with a minimal reproduction or a well-argued proposal.",
    ),
];

/// A value with how sure the strategy is about it and why.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Field<T> {
    pub value: T,
    /// 0..=1. Calibrated for decision models; self-reported for LLMs.
    #[serde(default)]
    pub confidence: Option<f64>,
    /// A short quote from the issue, or the reason for the value.
    #[serde(default)]
    pub evidence: Option<String>,
}

impl<T> Field<T> {
    pub fn new(value: T) -> Self {
        Field {
            value,
            confidence: None,
            evidence: None,
        }
    }

    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = Some(confidence);
        self
    }

    pub fn with_evidence(mut self, evidence: impl Into<String>) -> Self {
        self.evidence = Some(evidence.into());
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Bug,
    Feature,
    Question,
    Docs,
    Cleanup,
    FailingTest,
    Flake,
    Other,
    Spam,
}

impl Kind {
    pub const ALL: [Kind; 9] = [
        Kind::Bug,
        Kind::Feature,
        Kind::Question,
        Kind::Docs,
        Kind::Cleanup,
        Kind::FailingTest,
        Kind::Flake,
        Kind::Other,
        Kind::Spam,
    ];

    /// The serialized name.
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Bug => "bug",
            Kind::Feature => "feature",
            Kind::Question => "question",
            Kind::Docs => "docs",
            Kind::Cleanup => "cleanup",
            Kind::FailingTest => "failing_test",
            Kind::Flake => "flake",
            Kind::Other => "other",
            Kind::Spam => "spam",
        }
    }

    /// What the kind means, for prompts and decision-model options.
    pub fn description(self) -> &'static str {
        match self {
            Kind::Bug => "Something is broken or behaves incorrectly",
            Kind::Feature => "A request for new functionality or an enhancement",
            Kind::Question => "A support request or usage question, not a defect",
            Kind::Docs => "Missing, wrong or unclear documentation",
            Kind::Cleanup => "Refactoring, technical debt, dependency bumps, code health",
            Kind::FailingTest => "A CI test that fails consistently",
            Kind::Flake => "A CI test that fails intermittently",
            Kind::Other => "None of the above",
            Kind::Spam => "Spam, gibberish, or not a real issue",
        }
    }

    pub fn parse(s: &str) -> Option<Kind> {
        Kind::ALL.into_iter().find(|k| k.as_str() == s)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Critical,
    ImportantSoon,
    ImportantLongterm,
    Backlog,
}

impl Severity {
    pub const ALL: [Severity; 4] = [
        Severity::Critical,
        Severity::ImportantSoon,
        Severity::ImportantLongterm,
        Severity::Backlog,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Severity::Critical => "critical",
            Severity::ImportantSoon => "important_soon",
            Severity::ImportantLongterm => "important_longterm",
            Severity::Backlog => "backlog",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Severity::Critical => {
                "Must be fixed now: outage, data loss, security hole, release blocker"
            }
            Severity::ImportantSoon => "Should be fixed in the current release cycle",
            Severity::ImportantLongterm => "Important, but can wait a release or two",
            Severity::Backlog => "Nice to have; no urgency",
        }
    }

    pub fn parse(s: &str) -> Option<Severity> {
        Severity::ALL.into_iter().find(|k| k.as_str() == s)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Scope {
    /// One problem or request.
    Single,
    /// Several unrelated problems or requests bundled together.
    Multiple,
}

/// Whether a bug report has what a maintainer needs to reproduce it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Reproducibility {
    pub has_version: bool,
    pub has_environment: bool,
    pub has_steps: bool,
    pub has_expected_actual: bool,
    pub has_minimal_repro: bool,
    pub has_logs: bool,
    /// The checklist items that are missing.
    pub missing: Vec<ReproItem>,
    /// The overall verdict: can a maintainer start reproducing it?
    pub enough_info: bool,
}

impl Reproducibility {
    /// Builds the checklist, deriving `missing` from the flags.
    pub fn from_checks(
        has_version: bool,
        has_environment: bool,
        has_steps: bool,
        has_expected_actual: bool,
        has_minimal_repro: bool,
        has_logs: bool,
        enough_info: bool,
    ) -> Self {
        let missing = [
            (has_version, ReproItem::Version),
            (has_environment, ReproItem::Environment),
            (has_steps, ReproItem::Steps),
            (has_expected_actual, ReproItem::ExpectedActual),
            (has_minimal_repro, ReproItem::MinimalRepro),
            (has_logs, ReproItem::Logs),
        ]
        .into_iter()
        .filter_map(|(present, item)| (!present).then_some(item))
        .collect();
        Reproducibility {
            has_version,
            has_environment,
            has_steps,
            has_expected_actual,
            has_minimal_repro,
            has_logs,
            missing,
            enough_info,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ReproItem {
    Version,
    Environment,
    Steps,
    ExpectedActual,
    MinimalRepro,
    Logs,
}
