use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::config_hash;

/// The input of every strategy.
///
/// `config` is kept as raw JSON on the wire: [`TriageRequest::typed`] parses
/// it into the strategy's config type and hashes it as it was sent, so the
/// harness can compute the same hash without knowing the config's defaults.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TriageRequest<C = serde_json::Value> {
    pub config: C,
    pub issue: Issue,
    pub repo: Repo,
}

impl TriageRequest<serde_json::Value> {
    /// Parses the config into `C` and returns it with the config's hash.
    pub fn typed<C: DeserializeOwned>(
        self,
    ) -> Result<(TriageRequest<C>, String), serde_json::Error> {
        let hash = config_hash(&self.config);
        let config = serde_json::from_value(self.config)?;
        Ok((
            TriageRequest {
                config,
                issue: self.issue,
                repo: self.repo,
            },
            hash,
        ))
    }
}

/// The issue as it was when it was filed, trimmed to what a triager sees.
///
/// Labels, state, comments and anything else that reveals how the issue was
/// triaged are left out on purpose: they are the expected answers.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Issue {
    pub number: u64,
    pub title: String,
    /// The body with slash commands (`/kind bug`, `/sig node`, ...) removed.
    pub body: String,
    /// GitHub's author association: `NONE`, `CONTRIBUTOR`, `MEMBER`, ...
    pub author_association: String,
    /// RFC 3339.
    pub created_at: String,
}

/// The repository as it was when the issue was filed.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Repo {
    /// `owner/name`.
    pub name: String,
    /// The labels a triager can choose from.
    pub labels: Vec<Label>,
    /// The latest release when the issue was filed.
    #[serde(default)]
    pub latest_release: Option<String>,
    /// RFC 3339. Tools only see issues created before this.
    pub as_of: String,
}

impl Repo {
    /// The labels with the given prefix (`sig/`, `kind/`, ...).
    pub fn labels_with_prefix<'a>(
        &'a self,
        prefix: &'a str,
    ) -> impl Iterator<Item = &'a Label> + 'a {
        self.labels
            .iter()
            .filter(move |l| l.name.starts_with(prefix))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Label {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
}
