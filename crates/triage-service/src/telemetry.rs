use opentelemetry::global;
use tracing::field::{Field, Visit};
use tracing::level_filters::LevelFilter;
use tracing::span::{Attributes, Record};
use tracing::{Event, Id, Metadata, Subscriber};
use tracing_subscriber::layer::{Context, Filter};
use tracing_subscriber::registry::{LookupSpan, SpanRef};
use tracing_subscriber::{Layer, layer::SubscriberExt, util::SubscriberInitExt};

/// Builds the subscriber by hand so the replay filter can sit on the Logfire
/// layer too: a replayed handler must not export its events twice. Without
/// `LOGFIRE_TOKEN` nothing is exported and only the console output remains.
pub fn init(service_name: &str) -> anyhow::Result<logfire::ShutdownGuard> {
    // An empty `LOGFIRE_TOKEN=` (as in `.env.example`) counts as a token for
    // `IfTokenPresent`, which then exports without credentials and fails.
    let has_token = std::env::var("LOGFIRE_TOKEN").is_ok_and(|t| !t.trim().is_empty());
    let send = if has_token {
        logfire::config::SendToLogfire::IfTokenPresent
    } else {
        logfire::config::SendToLogfire::No
    };
    let logfire = logfire::configure()
        .local()
        .with_service_name(service_name)
        .send_to_logfire(send)
        .with_default_level_filter(LevelFilter::INFO)
        .finish()?;
    let guard = logfire.clone().shutdown_guard();

    global::set_text_map_propagator(opentelemetry_sdk::propagation::TraceContextPropagator::new());
    tracing_subscriber::registry()
        .with(logfire.tracing_layer().with_filter(ReplayAwareFilter))
        .with(
            tracing_subscriber::fmt::layer()
                .with_target(false)
                .with_filter(ReplayAwareFilter)
                .with_filter(LevelFilter::INFO),
        )
        .init();
    Ok(guard)
}

/// The span restate-sdk wraps every invocation attempt in.
const ENDPOINT_SPAN: &str = "restate_sdk_endpoint_handle";
/// The field restate-sdk records the replay state in. Spans that are current
/// while handler code calls `ctx` must declare it (empty), see below.
pub const REPLAYING_FIELD: &str = "restate.sdk.is_replaying";

/// Drops events logged while an invocation attempt replays its journal, like
/// `restate_sdk::filter::ReplayAwareFilter`, but without losing every event
/// after a retry.
///
/// restate-sdk 0.12.1 records the replay state with
/// `Span::current().record("restate.sdk.is_replaying", ..)`. When an attempt
/// starts replaying, the current span is its own endpoint span; when replay
/// ends, it is whatever span handler code is in, i.e. our `triage.run`
/// handler span. The SDK's filter only reads the field off the endpoint span,
/// so it never sees replay end and drops the rest of the attempt's events.
///
/// This filter takes the field from whichever span it is recorded on and
/// stores it on that span's endpoint span, where `event_enabled` looks. A
/// span only accepts the record if it declares the field, which is why
/// [`crate::trace::handler_span`] declares it. (Model and tool spans live
/// inside `ctx.run` closures, which never call back into `ctx`, so they don't
/// need it.)
pub struct ReplayAwareFilter;

#[derive(Debug, Clone, Copy)]
struct Replaying(bool);

#[derive(Default)]
struct ReplayingVisitor(Option<bool>);

impl Visit for ReplayingVisitor {
    fn record_bool(&mut self, field: &Field, value: bool) {
        if field.name() == REPLAYING_FIELD {
            self.0 = Some(value);
        }
    }

    fn record_debug(&mut self, _field: &Field, _value: &dyn std::fmt::Debug) {}
}

/// Stores the replay state on the nearest endpoint span at or above `span`.
fn store<S>(span: SpanRef<'_, S>, replaying: bool)
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    if let Some(endpoint) = span.scope().find(|s| s.name() == ENDPOINT_SPAN) {
        endpoint.extensions_mut().replace(Replaying(replaying));
    }
}

impl<S: Subscriber + for<'a> LookupSpan<'a>> Filter<S> for ReplayAwareFilter {
    fn enabled(&self, _meta: &Metadata<'_>, _cx: &Context<'_, S>) -> bool {
        true
    }

    fn event_enabled(&self, event: &Event<'_>, cx: &Context<'_, S>) -> bool {
        let Some(scope) = cx.event_scope(event) else {
            return true;
        };
        for span in scope {
            if span.name() == ENDPOINT_SPAN {
                return !span.extensions().get::<Replaying>().is_some_and(|r| r.0);
            }
        }
        true
    }

    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let mut visitor = ReplayingVisitor::default();
        attrs.record(&mut visitor);
        if let (Some(replaying), Some(span)) = (visitor.0, ctx.span(id)) {
            store(span, replaying);
        }
    }

    fn on_record(&self, id: &Id, values: &Record<'_>, ctx: Context<'_, S>) {
        let mut visitor = ReplayingVisitor::default();
        values.record(&mut visitor);
        if let (Some(replaying), Some(span)) = (visitor.0, ctx.span(id)) {
            store(span, replaying);
        }
    }
}

impl<S: Subscriber> Layer<S> for ReplayAwareFilter {}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use restate_sdk::context::HeaderMap;
    use tracing::{info, info_span};
    use tracing_subscriber::layer::Context;

    use super::*;

    /// Collects the messages of the events it is shown.
    #[derive(Clone, Default)]
    struct Collect(Arc<Mutex<Vec<String>>>);

    impl<S: Subscriber> Layer<S> for Collect {
        fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
            struct Message(String);
            impl Visit for Message {
                fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
                    if field.name() == "message" {
                        self.0 = format!("{value:?}");
                    }
                }
            }
            let mut m = Message(String::new());
            event.record(&mut m);
            self.0.lock().unwrap().push(m.0);
        }
    }

    /// What restate-sdk does in a retried attempt: the endpoint span starts
    /// with the field false and gets true while the journal is read; handler
    /// code runs in our handler span, and the SDK records the end of replay
    /// on that span (the current one), not on the endpoint span.
    fn retried_attempt<F>(filter: F) -> Vec<String>
    where
        F: Filter<tracing_subscriber::Registry> + Send + Sync + 'static,
    {
        let seen = Collect::default();
        let subscriber = tracing_subscriber::registry().with(seen.clone().with_filter(filter));
        tracing::subscriber::with_default(subscriber, || {
            let endpoint = info_span!(
                "restate_sdk_endpoint_handle",
                "restate.sdk.is_replaying" = false
            );
            let _e = endpoint.enter();
            info!("before replay");
            tracing::Span::current().record(REPLAYING_FIELD, true);
            info!("replay starts");
            let handler = crate::trace::handler_span("Test", "key", &HeaderMap::default());
            let _h = handler.enter();
            info!("replayed handler code");
            tracing::Span::current().record(REPLAYING_FIELD, false);
            info!("live handler code");
            drop(_h);
            info!("live endpoint code");
        });
        let out: Vec<String> = seen.0.lock().unwrap().clone();
        out
    }

    #[test]
    fn events_after_replay_are_kept() {
        assert_eq!(
            retried_attempt(ReplayAwareFilter),
            ["before replay", "live handler code", "live endpoint code"]
        );
    }

    /// The bug this filter works around: the SDK's own filter never sees
    /// replay end and drops everything after it.
    #[test]
    fn sdk_filter_drops_events_after_replay() {
        assert_eq!(
            retried_attempt(restate_sdk::filter::ReplayAwareFilter),
            ["before replay"]
        );
    }

    /// Events outside any invocation pass.
    #[test]
    fn events_outside_invocations_pass() {
        let seen = Collect::default();
        let subscriber =
            tracing_subscriber::registry().with(seen.clone().with_filter(ReplayAwareFilter));
        tracing::subscriber::with_default(subscriber, || {
            info!("plain");
            let _s = info_span!("other").entered();
            info!("in a span");
        });
        assert_eq!(*seen.0.lock().unwrap(), ["plain", "in a span"]);
    }
}
