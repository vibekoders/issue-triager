//! The hosted chat models of the experiment profiles accept the requests the
//! strategies send: each provider encodes them (structured output with the
//! strategies' answer schemas, function tools, the defaults `models` applies)
//! without refusing anything. No network: validation stops before sending,
//! and the API keys are placeholders unless real ones are set. Whether the
//! model then answers well, or at all, takes a live call.

use schemars::JsonSchema;
use triage_service::models::{self, ChatRequest};
use triage_service::strategies::agent::AgentAnswer;
use triage_service::strategies::agent_loop::decide_definition;
use triage_service::strategies::draft_verify::Verified;
use triage_service::strategies::single_shot::ModelTriage;
use triage_service::strategies::workflow::SubAgentAnswer;
use triage_service::tools;

/// The chat models of the hosted profiles in evals/experiments/profiles.yaml.
const MODELS: &[&str] = &[
    "openai:gpt-6-luna",
    "openai:gpt-6-sol",
    "anthropic:claude-haiku-4-5",
    "anthropic:claude-sonnet-5",
    "gemini:gemini-3.5-flash-lite",
    "gemini:gemini-3.1-pro-preview",
    "azure:gpt-6-luna",
    "azure:gpt-6-sol",
];

fn placeholder_credentials() {
    for (var, value) in [
        ("OPENAI_API_KEY", "sk-placeholder"),
        ("ANTHROPIC_API_KEY", "sk-ant-placeholder"),
        ("GEMINI_API_KEY", "placeholder"),
        ("AZURE_OPENAI_RESOURCE", "placeholder"),
        ("AZURE_OPENAI_API_KEY", "placeholder"),
    ] {
        if std::env::var_os(var).is_none_or(|v| v.is_empty()) {
            // SAFETY: this test binary has a single test, so nothing reads
            // the environment concurrently.
            unsafe { std::env::set_var(var, value) };
        }
    }
}

/// What `allama::extract::Extractor` sends for `T` in native mode, the mode
/// it picks for every model here (structured output `Supported`, or
/// `Unknown` for GPT-6, which it tries natively first).
fn extraction<T: JsonSchema>(model: &str) -> ChatRequest {
    let schema = serde_json::to_value(schemars::schema_for!(T)).unwrap();
    ChatRequest::new(model)
        .system("You triage GitHub issues.")
        .user("Title: kubelet crashes\n\nIt crashes.")
        .max_output_tokens(6000)
        .response_format(allama::ResponseFormat::json_schema(schema).name("answer"))
}

#[test]
fn hosted_models_accept_the_strategy_requests() {
    placeholder_credentials();
    let mut failures = Vec::new();
    for model in MODELS {
        let mut tool_defs = tools::definitions();
        tool_defs.push(decide_definition());
        let requests = [
            ("ModelTriage", extraction::<ModelTriage>(model)),
            ("AgentAnswer", extraction::<AgentAnswer>(model)),
            ("Verified", extraction::<Verified>(model)),
            ("SubAgentAnswer", extraction::<SubAgentAnswer>(model)),
            (
                "agent turn with tools",
                ChatRequest::new(*model)
                    .system("You triage GitHub issues.")
                    .user("Find duplicates of this issue.")
                    .max_output_tokens(4000)
                    .tools(tool_defs),
            ),
        ];
        for (what, request) in requests {
            if let Err(e) = models::validate(request) {
                failures.push(format!("{model} {what}: {e}"));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
