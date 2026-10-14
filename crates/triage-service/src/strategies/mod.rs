//! One Restate workflow per strategy, each with a `run` handler that takes a
//! `TriageRequest` and returns a `TriageResponse`.
//!
//! The workflow key is `{run_id}:{config_hash}:{case_digest}:{repeat}`, chosen
//! by the harness. A workflow runs once per key, so re-posting a case returns
//! the stored result instead of paying for the models again.
//!
//! # Restate timeouts
//!
//! Every strategy sets `inactivity_timeout = "5m"` and `abort_timeout = "1h"`
//! on its workflow (docker-compose.yaml sets the same server-wide defaults).
//! A single model call on a loaded CPU can take many minutes, and an
//! in-flight `ctx.run` sends nothing to Restate meanwhile. With Restate's
//! defaults (1 minute of inactivity, then a 10 minute abort timeout) such a
//! call is cut off and started over on every retry, paid for each time but
//! counted once in `Usage`. Heuristic makes no calls but gets the same
//! settings, so every strategy behaves alike.

pub mod agent;
pub mod agent_loop;
pub mod decision_triage;
pub mod draft_verify;
pub mod ensemble;
pub mod heuristic;
pub mod info;
mod sig_tokens;
pub mod single_shot;
pub mod workflow;

use restate_sdk::prelude::*;
use serde::de::DeserializeOwned;
use triage_core::{Triage, TriageRequest, TriageResponse, Usage};

pub use agent::AgentStrategy;
pub use draft_verify::DraftVerifyStrategy;
pub use ensemble::EnsembleStrategy;
pub use heuristic::HeuristicStrategy;
pub use info::InfoService;
pub use single_shot::SingleShotStrategy;
pub use workflow::WorkflowStrategy;

/// Parses the request's config into the strategy's config type. A config
/// that doesn't parse fails the same way on every retry, so it is terminal.
pub fn parse_config<C: DeserializeOwned>(
    request: TriageRequest,
) -> Result<(TriageRequest<C>, String), TerminalError> {
    request
        .typed::<C>()
        .map_err(|e| TerminalError::new_with_code(400, format!("invalid config: {e}")))
}

/// Records the strategy version and config hash on the handler span, which
/// [`crate::trace::handler_span`] declared empty.
pub fn record_run(span: &tracing::Span, version: &str, config_hash: &str) {
    span.record("strategy.version", version);
    span.record("config_hash", config_hash);
}

pub fn response(
    strategy: &str,
    version: &str,
    config_hash: String,
    triage: Triage,
    usage: Usage,
) -> TriageResponse {
    TriageResponse {
        triage,
        usage,
        strategy: strategy.to_owned(),
        version: version.to_owned(),
        config_hash,
    }
}

/// For the strategies that are not written yet.
pub fn not_implemented(strategy: &str) -> HandlerError {
    TerminalError::new_with_code(501, format!("{strategy}: not implemented yet")).into()
}
