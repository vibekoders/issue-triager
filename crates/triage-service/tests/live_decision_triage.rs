//! A live check of the decision triage against a local Ollama. Ignored by
//! default:
//!
//! ```text
//! cargo test -p triage-service --test live_decision_triage -- --ignored --nocapture
//! ```
//!
//! `LIVE_DECISION_MODEL` (default `ollama:tev1:0.8b-q8_0`) picks the model.
//! `LIVE_CASE` can point at a JSON file `{"issue": ..., "repo": ...}` (a
//! dataset case's inputs); without it a small built-in issue is used.

use std::time::Instant;

use triage_core::config::Thresholds;
use triage_core::{Issue, Label, Repo};
use triage_service::strategies::decision_triage;

fn decision_model() -> String {
    std::env::var("LIVE_DECISION_MODEL").unwrap_or_else(|_| "ollama:tev1:0.8b-q8_0".into())
}

fn case() -> (Issue, Repo) {
    if let Ok(path) = std::env::var("LIVE_CASE") {
        #[derive(serde::Deserialize)]
        struct Case {
            issue: Issue,
            repo: Repo,
        }
        let case: Case = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        return (case.issue, case.repo);
    }
    let labels = [
        "sig/node",
        "sig/network",
        "sig/storage",
        "sig/api-machinery",
    ]
    .into_iter()
    .map(|name| Label {
        name: name.into(),
        description: Some(format!("Categorizes an issue or PR as relevant to {name}.")),
    })
    .collect();
    (
        Issue {
            number: 1,
            title: "kubelet OOMKilled pods after upgrading to v1.32.1".into(),
            body: "### What happened?\n\nAfter upgrading from v1.31.4 to v1.32.1 the kubelet \
                   kills pods with OOMKilled although they use far less memory than their \
                   limit.\n\n### What did you expect to happen?\n\nPods keep running.\n\n\
                   ### How can we reproduce it?\n\nkubectl apply -f pod.yaml with a 512Mi limit.\n\n\
                   ### Kubernetes version\n\nv1.32.1\n"
                .into(),
            author_association: "NONE".into(),
            created_at: "2025-03-01T12:00:00Z".into(),
        },
        Repo {
            name: "kubernetes/kubernetes".into(),
            labels,
            latest_release: Some("v1.32.2".into()),
            as_of: "2025-03-01T12:00:00Z".into(),
        },
    )
}

#[tokio::test]
#[ignore]
async fn decision_triage_answers_every_field() {
    let (issue, repo) = case();
    let started = Instant::now();
    let summary = decision_triage::decide(&decision_model(), &issue, &repo)
        .await
        .unwrap();
    let elapsed = started.elapsed();
    println!("elapsed: {elapsed:?}\nusage: {:?}", summary.usage);
    for (name, answer) in &summary.answers {
        println!("  {name}: {answer:?}");
    }
    let triage = decision_triage::to_triage(&summary, &repo);
    println!("{}", serde_json::to_string_pretty(&triage).unwrap());
    let low = decision_triage::low_confidence_fields(&triage, &Thresholds::default());
    println!("below 0.8: {low:?}");
    // One call, or one per question group when the model's context is short.
    assert!((1..=3).contains(&summary.usage.decision_calls));
    assert!(triage.kind.is_some() && triage.is_regression.is_some());
    assert!(triage.quality.is_some() && triage.scope.is_some());
}

/// The agent loop's `decide` tool, answered over the same input.
#[tokio::test]
#[ignore]
async fn decide_tool() {
    use triage_service::strategies::agent_loop::{self, DecisionTool};
    let (issue, repo) = case();
    let tool = DecisionTool {
        model: decision_model(),
        input: decision_triage::input(&issue, &repo),
    };
    let args = r#"{"question": "Which component is affected?", "kind": "choice",
                  "options": {"kubelet": "the node agent", "kubectl": "the CLI", "apiserver": "the API server"}}"#;
    let out = agent_loop::decide(&tool, args).await.unwrap();
    println!("{out:?}");
    assert!(!out.is_error, "{}", out.content);
    assert_eq!(out.usage.decision_calls, 1);
    assert_eq!(out.usage.tool_calls, 0);
}
