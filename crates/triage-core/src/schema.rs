//! The JSON schemas of the request, response and config types, which the
//! `triage-schema` binary writes into `schema/` for the Python harness.

use schemars::{Schema, schema_for};

use crate::config::*;
use crate::{TriageRequest, TriageResponse};

/// Every schema, as `(name, schema)`; the file is `{name}.schema.json`.
pub fn schemas() -> Vec<(&'static str, Schema)> {
    vec![
        ("triage-request", schema_for!(TriageRequest)),
        ("triage-response", schema_for!(TriageResponse)),
        ("heuristic-config", schema_for!(HeuristicConfig)),
        ("single-shot-config", schema_for!(SingleShotConfig)),
        ("workflow-config", schema_for!(WorkflowConfig)),
        ("agent-config", schema_for!(AgentConfig)),
        ("ensemble-config", schema_for!(EnsembleConfig)),
        ("draft-verify-config", schema_for!(DraftVerifyConfig)),
    ]
}

/// A schema file's contents: pretty JSON with a trailing newline.
pub fn render(schema: &Schema) -> String {
    serde_json::to_string_pretty(schema).expect("a schema serializes") + "\n"
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    /// The committed `schema/` must match the Rust types, so the harness never
    /// validates against a stale schema.
    #[test]
    fn committed_schemas_are_up_to_date() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../schema");
        let mut stale = Vec::new();
        for (name, schema) in schemas() {
            let path = dir.join(format!("{name}.schema.json"));
            let committed = std::fs::read_to_string(&path).unwrap_or_default();
            if committed != render(&schema) {
                stale.push(path.display().to_string());
            }
        }
        assert!(
            stale.is_empty(),
            "out of date with the Rust types: {}; run `cargo run -p triage-core --bin triage-schema`",
            stale.join(", ")
        );
    }
}
