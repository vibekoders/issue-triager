//! The triage service: one Restate workflow per strategy, the model and tool
//! calls they share, and the telemetry that ties their spans to the eval
//! harness's traces.

pub mod models;
pub mod strategies;
pub mod telemetry;
pub mod tools;
pub mod trace;

use restate_sdk::prelude::*;

/// The Restate endpoint with every strategy bound, plus the `Info` service.
pub fn endpoint() -> Endpoint {
    use strategies::*;
    Endpoint::builder()
        .bind(HeuristicStrategy)
        .bind(SingleShotStrategy)
        .bind(WorkflowStrategy)
        .bind(AgentStrategy)
        .bind(EnsembleStrategy)
        .bind(DraftVerifyStrategy)
        .bind(InfoService)
        .build()
}
