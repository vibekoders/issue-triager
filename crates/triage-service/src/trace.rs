//! Trace-context plumbing between the harness, Restate and the handlers.

use std::collections::HashMap;

use opentelemetry::global;
use opentelemetry::propagation::Extractor;
use restate_sdk::context::{HeaderMap, Request};
use tracing::{field::Empty, info_span};
use tracing_opentelemetry::OpenTelemetrySpanExt;

/// Restate's handler headers are `http::HeaderMap<String>`.
struct RestateHeaders<'a>(&'a HeaderMap);

impl Extractor for RestateHeaders<'_> {
    fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).map(String::as_str)
    }

    fn keys(&self) -> Vec<&str> {
        self.0.keys().map(|k| k.as_str()).collect()
    }
}

const TRACE_HEADERS: [&str; 2] = ["traceparent", "tracestate"];

/// The trace context the caller sent with this invocation. Restate stores the
/// headers with the invocation, so it is the same on every attempt.
///
/// Restate doesn't put its own span context in these headers, and calls made
/// through `ctx` carry no headers unless the caller adds them.
pub fn parent_context(headers: &HeaderMap) -> opentelemetry::Context {
    extract_context(&RestateHeaders(headers))
}

/// Like `TextMapPropagator::extract`, but without falling back to the current
/// context, which inside a handler is the SDK's `restate_sdk_endpoint_handle`
/// span: that span is new on every attempt, so anything parented on it would
/// land in a different trace each time. With no incoming context the spans
/// below start a new trace instead.
pub fn extract_context(extractor: &dyn Extractor) -> opentelemetry::Context {
    global::get_text_map_propagator(|p| {
        p.extract_with_context(&opentelemetry::Context::new(), extractor)
    })
}

/// The trace-context headers of the current invocation, to forward to calls.
pub fn incoming_trace_headers(headers: &HeaderMap) -> HashMap<String, String> {
    TRACE_HEADERS
        .iter()
        .filter_map(|&k| headers.get(k).map(|v| (k.to_owned(), v.clone())))
        .collect()
}

/// Adds the trace-context headers to a call. Restate journals the call with
/// its headers and checks them on replay, so they must be added in the same
/// order every time: hence the fixed `TRACE_HEADERS` order rather than the
/// map's. Empty values (an unset `tracestate`) are left out.
pub fn with_trace_headers<'a, Req, Res>(
    mut req: Request<'a, Req, Res>,
    headers: &HashMap<String, String>,
) -> Request<'a, Req, Res> {
    for k in TRACE_HEADERS {
        if let Some(v) = headers.get(k).filter(|v| !v.is_empty()) {
            req = req.header(k.to_owned(), v.clone());
        }
    }
    req
}

/// Creates the handler's root span and parents it on the incoming
/// `traceparent`, so the workflow's spans land in the eval's trace. Must run
/// before the span is first entered.
///
/// It declares restate-sdk's replay field: the SDK records the end of replay
/// on the current span, which is this one, and
/// [`crate::telemetry::ReplayAwareFilter`] picks it up from here.
pub fn handler_span(strategy: &'static str, eval_key: &str, headers: &HeaderMap) -> tracing::Span {
    let span = info_span!(
        "triage.run",
        "eval.key" = eval_key,
        "strategy" = strategy,
        "strategy.version" = Empty,
        "config_hash" = Empty,
        "trace.linked" = Empty,
        "restate.sdk.is_replaying" = Empty,
    );
    let linked =
        span.set_parent(parent_context(headers)).is_ok() && headers.contains_key("traceparent");
    span.record("trace.linked", linked);
    span
}
