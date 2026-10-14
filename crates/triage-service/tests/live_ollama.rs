//! Live checks of the model calls against a local Ollama. Ignored by default:
//!
//! ```text
//! cargo test -p triage-service --test live_ollama -- --ignored --nocapture
//! ```
//!
//! `LIVE_CHAT_MODEL` (default `ollama:qwen3:1.7b`) and `LIVE_DECISION_MODEL`
//! (default `ollama:tev1:0.8b-q8_0`) pick the models.

use schemars::JsonSchema;
use serde::Deserialize;
use triage_service::models::{self, ChatRequest, DecisionAnswer, DecisionRequest, Question};
use triage_service::tools;

fn chat_model() -> String {
    std::env::var("LIVE_CHAT_MODEL").unwrap_or_else(|_| "ollama:qwen3:1.7b".into())
}

fn decision_model() -> String {
    std::env::var("LIVE_DECISION_MODEL").unwrap_or_else(|_| "ollama:tev1:0.8b-q8_0".into())
}

#[derive(Debug, Deserialize, JsonSchema)]
struct Label {
    #[schemars(extend("enum" = ["bug", "feature"]))]
    kind: String,
}

#[tokio::test]
#[ignore]
async fn structured_output() {
    let request = ChatRequest::new(chat_model())
        .user("Classify this GitHub issue as bug or feature: \"kubelet panics on startup\"");
    let out = models::extract::<Label>("live_test", request)
        .await
        .unwrap();
    println!("{out:?}");
    assert_eq!(out.value.kind, "bug");
    assert_eq!(out.usage.llm_calls, 1);
    assert!(out.usage.input_tokens > 0 && out.usage.output_tokens > 0);
    assert!(out.usage.cost_usd > 0.0);
}

/// One agent-loop round trip: the model asks for a tool, gets the result,
/// then answers.
#[tokio::test]
#[ignore]
async fn tool_round_trip() {
    let mut request = ChatRequest::new(chat_model())
        .system("You triage Kubernetes issues. Use the tools when they help.")
        .user("Which SIG labels does this repository have? Call list_labels, then answer with the label names only.")
        .tools(tools::definitions_for(&[tools::LIST_LABELS]));
    let turn = models::chat("live_test", request.clone()).await.unwrap();
    println!("{turn:?}");
    assert_eq!(turn.tool_calls.len(), 1, "expected a tool call: {turn:?}");

    let repo = triage_core::Repo {
        name: "kubernetes/kubernetes".into(),
        labels: vec![triage_core::Label {
            name: "sig/node".into(),
            description: None,
        }],
        latest_release: None,
        as_of: "2025-01-01T00:00:00Z".into(),
    };
    let ctx = tools::ToolContext::new(repo, u64::MAX);
    request.messages.push(turn.message.clone());
    for call in &turn.tool_calls {
        let out = tools::call(&ctx, &call.name, &call.arguments)
            .await
            .unwrap();
        assert_eq!(out.usage.tool_calls, 1);
        request.push_tool_result(call, out.content);
    }
    let answer = models::chat("live_test", request).await.unwrap();
    println!("{answer:?}");
    assert!(answer.tool_calls.is_empty());
    assert!(answer.text.unwrap_or_default().contains("sig/node"));
}

#[tokio::test]
#[ignore]
async fn decision() {
    let request = DecisionRequest::new(
        decision_model(),
        "kubelet panics with a nil pointer dereference on startup after upgrading to v1.33",
    )
    .question(
        "kind",
        Question::choice("What kind of GitHub issue is this?")
            .option("bug", "Something is broken")
            .option("feature", "A request for new functionality"),
    )
    .question(
        "regression",
        Question::yes_no("Did it work in an earlier version?"),
    );
    let out = models::decide("live_test", request).await.unwrap();
    println!("{out:#?}");
    assert_eq!(out.usage.decision_calls, 1);
    assert!(matches!(
        out.answers.get("kind"),
        Some(DecisionAnswer::Choice { .. })
    ));
    assert!(out.yes_no("regression").is_some());
}
