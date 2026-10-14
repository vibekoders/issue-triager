//! `triage-service [serve]`: serves the Restate endpoint on `LISTEN_ADDR`
//! (default `0.0.0.0:9080`).

use restate_sdk::prelude::*;
use tracing::info;

async fn serve() -> anyhow::Result<()> {
    let addr = std::env::var("LISTEN_ADDR").unwrap_or_else(|_| "0.0.0.0:9080".into());
    let addr = addr.parse()?;
    info!(%addr, "serving Restate endpoint");
    // Return on SIGINT or SIGTERM, so `main` flushes the last batch of spans.
    let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    tokio::select! {
        _ = HttpServer::new(triage_service::endpoint()).listen_and_serve(addr) => {}
        _ = tokio::signal::ctrl_c() => info!("SIGINT, shutting down"),
        _ = sigterm.recv() => info!("SIGTERM, shutting down"),
    }
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    match std::env::args().nth(1).as_deref() {
        None | Some("serve") => {}
        Some(other) => anyhow::bail!("unknown command {other:?}; usage: triage-service [serve]"),
    }
    let guard = triage_service::telemetry::init("triage-service")?;
    let result = serve().await;
    if let Err(e) = guard.shutdown() {
        eprintln!("warning: flushing traces to Logfire failed: {e}");
    }
    result
}
