//! `Info`: a plain Restate service describing the deployed strategies.
//!
//! The harness names experiments `{strategy}@{version} ...`, so it asks the
//! running service for the versions instead of keeping its own copy that
//! could drift from the code.

use std::collections::BTreeMap;

use restate_sdk::prelude::*;

use super::{agent, draft_verify, ensemble, heuristic, single_shot, workflow};

pub struct InfoService;

/// Each strategy's Restate service name with its code version.
pub fn versions() -> BTreeMap<String, String> {
    [
        ("Heuristic", heuristic::VERSION),
        ("SingleShot", single_shot::VERSION),
        ("Workflow", workflow::VERSION),
        ("Agent", agent::VERSION),
        ("Ensemble", ensemble::VERSION),
        ("DraftVerify", draft_verify::VERSION),
    ]
    .into_iter()
    .map(|(name, version)| (name.to_owned(), version.to_owned()))
    .collect()
}

#[restate_sdk::service(name = "Info")]
impl InfoService {
    /// `{"Heuristic": "1", "SingleShot": "1", ...}`.
    #[handler]
    async fn versions(
        &self,
        _ctx: Context<'_>,
    ) -> Result<Json<BTreeMap<String, String>>, HandlerError> {
        Ok(Json(versions()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lists_every_strategy() {
        let v = versions();
        assert_eq!(v.len(), 6);
        assert_eq!(v["Heuristic"], heuristic::VERSION);
        assert_eq!(v["DraftVerify"], draft_verify::VERSION);
    }
}
