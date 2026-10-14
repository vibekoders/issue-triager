# syntax=docker/dockerfile:1
#
# The triage service (crates/triage-service), so the strategies run without a
# Rust toolchain:
#
#   docker compose --profile app up -d --build
#
# The search corpus for the agent's tools is not in the image; mount it at
# /app/data/corpus (docker-compose.yaml does). API keys and LOGFIRE_TOKEN come
# from the environment (env_file: .env in docker-compose.yaml), never the image.

FROM rust:1.98-slim-trixie AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
# Cache mounts keep the registry and compiled dependencies between builds, so
# a change to the service rebuilds only the workspace crates.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked -p triage-service \
    && cp target/release/triage-service /usr/local/bin/triage-service

FROM debian:trixie-slim
# CA certificates for the hosted providers (TLS through rustls with the
# system's roots) and Logfire.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --system --uid 10001 --create-home --home-dir /app triage
COPY --from=build /usr/local/bin/triage-service /usr/local/bin/triage-service
USER triage
WORKDIR /app
ENV LISTEN_ADDR=0.0.0.0:9080 \
    CORPUS_DIR=/app/data/corpus
EXPOSE 9080
# Exec form: the service is PID 1 and gets SIGTERM from `docker stop`, which
# it handles by flushing the last spans to Logfire before exiting.
ENTRYPOINT ["triage-service"]
CMD ["serve"]
