//! Strategy configs. Each strategy's Restate service parses the request's
//! `config` into one of these.
//!
//! Models are addressed as `provider:model`, e.g. `ollama:qwen3:4b`,
//! `ollama:nimble`, `typesafe:jev-latest` or `anthropic:claude-sonnet-5`.

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// `Heuristic`: no models. Every answer comes from the issue's text, and a
/// field the text gives no signal for is left out.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct HeuristicConfig {
    /// Fill the fields the text gives no signal for with dev-split base
    /// rates instead (kind: bug; area: the most common SIG; severity: the
    /// most common priority for the kind), as version 2 always did. A
    /// separate prior-only baseline: the plain heuristic doesn't guess.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub base_rates: bool,
}

/// `SingleShot`: one LLM call with structured output.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SingleShotConfig {
    pub model: String,
    #[serde(default)]
    pub temperature: Option<f32>,
}

/// `Workflow`: one decision-model call, then LLM escalation for the fields
/// it's unsure about.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct WorkflowConfig {
    /// A System One decision model, e.g. `ollama:nimble` or `typesafe:jev-latest`.
    pub decision_model: String,
    /// `None` (or absent): decision model only.
    #[serde(default)]
    pub escalation: Option<Escalation>,
    /// Writes `suggested_reply` from the final triage. `None`: no reply.
    #[serde(default)]
    pub reply_model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Escalation {
    pub model: String,
    #[serde(default)]
    pub thresholds: Thresholds,
    /// Use a tool-using sub-agent (instead of a single LLM call) for the
    /// fields that need repo context: `area` and `duplicates`.
    #[serde(default)]
    pub sub_agent: bool,
}

/// Confidence thresholds for accepting a decision-model answer.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Thresholds {
    /// Answers at or above this confidence are accepted as they are.
    #[serde(default = "default_accept")]
    pub accept: f64,
    /// Per-field overrides of `accept`, by `Triage` field name.
    #[serde(default)]
    pub per_field: BTreeMap<String, f64>,
}

impl Thresholds {
    pub fn accept(&self, field: &str) -> f64 {
        self.per_field.get(field).copied().unwrap_or(self.accept)
    }
}

impl Default for Thresholds {
    fn default() -> Self {
        Thresholds {
            accept: default_accept(),
            per_field: BTreeMap::new(),
        }
    }
}

fn default_accept() -> f64 {
    0.8
}

/// `Agent`: a tool-using agent loop.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AgentConfig {
    pub model: String,
    /// Gives the agent a `decide` tool backed by this decision model.
    #[serde(default)]
    pub decision_tool: Option<String>,
    #[serde(default = "default_max_turns")]
    pub max_turns: u32,
    /// Which corpus tools the agent gets, by name: `search_issues`,
    /// `get_issue`, `list_labels`, `get_issue_templates`, `get_releases`.
    /// Absent: all of them. `[]`: none, so there is no loop and the agent is a
    /// single structured call (the `decide` tool, if configured, still counts
    /// as a tool).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<String>>,
}

fn default_max_turns() -> u32 {
    8
}

/// `Ensemble`: several voters, one aggregator.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct EnsembleConfig {
    pub voters: Vec<Voter>,
    #[serde(default)]
    pub aggregator: Aggregator,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Voter {
    /// A single-shot LLM call.
    Llm {
        model: String,
        #[serde(default)]
        temperature: Option<f32>,
    },
    /// A decision-model call. Its votes are weighted by their confidence.
    Decision { model: String },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Aggregator {
    /// Per-field (confidence-weighted) majority vote, in code.
    #[default]
    Vote,
    /// An LLM reads the issue and all votes and decides.
    Judge { model: String },
}

/// `DraftVerify`: a decision-model draft, then one agent verifies the fields
/// below the threshold.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DraftVerifyConfig {
    pub decision_model: String,
    pub verifier_model: String,
    #[serde(default)]
    pub thresholds: Thresholds,
    #[serde(default = "default_max_turns")]
    pub max_turns: u32,
}
