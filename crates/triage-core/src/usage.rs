use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// What a triage cost. Summed from journaled `ctx.run` results, so it is the
/// same on every replay.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub llm_calls: u64,
    pub decision_calls: u64,
    pub tool_calls: u64,
    /// Summed duration of the model and tool calls, not the wall time of the
    /// whole invocation (which includes Restate's own scheduling).
    pub latency_ms: u64,
    /// In USD. Local models are priced at shadow prices, see [`crate::price`].
    pub cost_usd: f64,
}

impl Usage {
    pub fn add(&mut self, other: &Usage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.llm_calls += other.llm_calls;
        self.decision_calls += other.decision_calls;
        self.tool_calls += other.tool_calls;
        self.latency_ms += other.latency_ms;
        self.cost_usd += other.cost_usd;
    }
}

impl std::ops::AddAssign<&Usage> for Usage {
    fn add_assign(&mut self, other: &Usage) {
        self.add(other);
    }
}
