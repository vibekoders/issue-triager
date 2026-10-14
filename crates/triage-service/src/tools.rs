//! Agent tools over the local issue corpus (`data/corpus/<owner>__<name>.sqlite`).
//!
//! The corpus is a snapshot, never live GitHub, and every query is cut off at
//! `repo.as_of`: a live search would find issues filed after the one being
//! triaged, including its own duplicates and the fix. Results never include
//! labels, state or close reasons, which are the expected answers.
//!
//! An earlier issue's title and body may still have been edited after
//! `as_of` (the corpus keeps original bodies only for eligible cases): a
//! maintainer lists the duplicates filed later, or adds "fixed by #N". So
//! every reference to an issue or pull request of this repo numbered at or
//! above the triaged issue (GitHub numbers both in one sequence, so those
//! were filed after it) is replaced by [`LATER_REF`], and search terms that
//! are such numbers are dropped: searching for its own number would
//! otherwise find exactly the issues edited to point at it.
//!
//! Tool failures (a bad argument, a missing corpus) are returned to the
//! model as an error message rather than failing the handler, so the agent
//! can recover or answer without the tool.

use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::time::Instant;

use allama::ToolDefinition;
use regex::{Captures, Regex};
use rusqlite::{Connection, OpenFlags, OptionalExtension, params};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tracing::{Instrument, field::Empty, info_span};
use triage_core::{Repo, Usage};

use crate::models::CallError;

pub const SEARCH_ISSUES: &str = "search_issues";
pub const GET_ISSUE: &str = "get_issue";
pub const LIST_LABELS: &str = "list_labels";
pub const GET_ISSUE_TEMPLATES: &str = "get_issue_templates";
pub const GET_RELEASES: &str = "get_releases";

const DEFAULT_SEARCH_LIMIT: u32 = 10;
const MAX_SEARCH_LIMIT: u32 = 25;
/// Longer bodies are cut, to keep a few tool results within a small model's
/// context window.
const MAX_BODY_CHARS: usize = 4000;
const MAX_QUERY_TERMS: usize = 16;
const MAX_RELEASES: u32 = 20;

/// What replaces a reference to an issue or pull request filed after the
/// triaged issue.
pub const LATER_REF: &str = "[a later issue]";

/// What the tools need from the request.
#[derive(Debug, Clone)]
pub struct ToolContext {
    pub repo: Repo,
    /// The number of the issue being triaged: references to it and to
    /// anything numbered after it are redacted from results.
    pub issue_number: u64,
    pub corpus_dir: PathBuf,
}

impl ToolContext {
    /// The corpus is looked up in `$CORPUS_DIR` (default `./data/corpus`).
    pub fn new(repo: Repo, issue_number: u64) -> Self {
        let corpus_dir = std::env::var_os("CORPUS_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("data/corpus"));
        Self::with_corpus_dir(repo, issue_number, corpus_dir)
    }

    pub fn with_corpus_dir(repo: Repo, issue_number: u64, corpus_dir: impl Into<PathBuf>) -> Self {
        ToolContext {
            repo,
            issue_number,
            corpus_dir: corpus_dir.into(),
        }
    }

    /// `<corpus_dir>/<owner>__<name>.sqlite`.
    pub fn corpus_path(&self) -> PathBuf {
        self.corpus_dir
            .join(format!("{}.sqlite", self.repo.name.replace('/', "__")))
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchIssuesArgs {
    /// Keywords to look for in issue titles and bodies, e.g. an error
    /// message or component names.
    pub query: String,
    /// How many results to return (default 10, at most 25).
    #[serde(default)]
    pub limit: Option<u32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetIssueArgs {
    /// The issue number.
    pub number: u64,
}

/// The tool definitions to offer a model, all of them.
pub fn definitions() -> Vec<ToolDefinition> {
    let no_args = json!({"type": "object", "properties": {}, "additionalProperties": false});
    vec![
        ToolDefinition::from_schema::<SearchIssuesArgs>(SEARCH_ISSUES).description(
            "Full-text search over the repository's earlier issues (filed before this one). \
             Returns issue numbers, titles and a snippet, best match first. Use it to find \
             duplicates or related reports.",
        ),
        ToolDefinition::from_schema::<GetIssueArgs>(GET_ISSUE).description(
            "Fetch an earlier issue of the repository by number: title, body and creation time.",
        ),
        ToolDefinition::new(LIST_LABELS, no_args.clone())
            .description("List the repository's triage labels with their descriptions."),
        ToolDefinition::new(GET_ISSUE_TEMPLATES, no_args.clone()).description(
            "Get the repository's issue templates (bug report, feature request, ...).",
        ),
        ToolDefinition::new(GET_RELEASES, no_args).description(
            "List the repository's most recent releases published before this issue, newest first.",
        ),
    ]
}

/// The definitions of the named tools, in the given order; unknown names
/// are skipped.
pub fn definitions_for(names: &[&str]) -> Vec<ToolDefinition> {
    let all = definitions();
    names
        .iter()
        .filter_map(|n| all.iter().find(|d| d.name == *n).cloned())
        .collect()
}

/// A tool result for the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOutput {
    /// JSON on success, an `error: ...` message otherwise.
    pub content: String,
    pub is_error: bool,
    pub usage: Usage,
}

/// Runs one tool call. Like the model calls, it records a counted span, so
/// run it through [`crate::models::journal`].
///
/// It never fails: problems become an error message for the model. The
/// `Result` is there to fit `journal`.
pub async fn call(ctx: &ToolContext, name: &str, arguments: &str) -> Result<ToolOutput, CallError> {
    let span = info_span!(
        "tool",
        "tool.name" = name,
        "tool.arguments" = arguments,
        "tool.is_error" = Empty,
        "tool.latency_ms" = Empty,
    );
    async move {
        let started = Instant::now();
        let ctx = ctx.clone();
        let (name_owned, args_owned) = (name.to_owned(), arguments.to_owned());
        let result =
            tokio::task::spawn_blocking(move || dispatch(&ctx, &name_owned, &args_owned)).await;
        let (content, is_error) = match result {
            Ok(Ok(value)) => (value.to_string(), false),
            Ok(Err(message)) => (format!("error: {message}"), true),
            Err(e) => (format!("error: the tool crashed: {e}"), true),
        };
        let latency_ms = started.elapsed().as_millis() as u64;
        let span = tracing::Span::current();
        span.record("tool.is_error", is_error);
        span.record("tool.latency_ms", latency_ms as i64);
        Ok(ToolOutput {
            content,
            is_error,
            usage: Usage {
                tool_calls: 1,
                latency_ms,
                ..Usage::default()
            },
        })
    }
    .instrument(span)
    .await
}

/// Runs a tool synchronously. `Err` is a message for the model.
pub fn dispatch(ctx: &ToolContext, name: &str, arguments: &str) -> Result<Value, String> {
    let arguments = if arguments.trim().is_empty() {
        "{}"
    } else {
        arguments
    };
    match name {
        SEARCH_ISSUES => {
            let args: SearchIssuesArgs = parse_args(arguments)?;
            let limit = args
                .limit
                .unwrap_or(DEFAULT_SEARCH_LIMIT)
                .clamp(1, MAX_SEARCH_LIMIT);
            search_issues(&open_corpus(ctx)?, ctx, &args.query, limit)
        }
        GET_ISSUE => {
            let args: GetIssueArgs = parse_args(arguments)?;
            get_issue(&open_corpus(ctx)?, ctx, args.number)
        }
        LIST_LABELS => Ok(json!(
            ctx.repo
                .labels
                .iter()
                .map(|l| json!({"name": l.name, "description": l.description}))
                .collect::<Vec<_>>()
        )),
        GET_ISSUE_TEMPLATES => get_issue_templates(&open_corpus(ctx)?),
        GET_RELEASES => get_releases(&open_corpus(ctx)?, &ctx.repo.as_of),
        other => Err(format!(
            "unknown tool `{other}`; available tools: {SEARCH_ISSUES}, {GET_ISSUE}, \
             {LIST_LABELS}, {GET_ISSUE_TEMPLATES}, {GET_RELEASES}"
        )),
    }
}

fn parse_args<T: serde::de::DeserializeOwned>(arguments: &str) -> Result<T, String> {
    serde_json::from_str(arguments).map_err(|e| format!("invalid arguments: {e}"))
}

fn open_corpus(ctx: &ToolContext) -> Result<Connection, String> {
    let path = ctx.corpus_path();
    open_read_only(&path).map_err(|e| {
        format!(
            "the issue corpus for {} is not available ({}: {e})",
            ctx.repo.name,
            path.display()
        )
    })
}

fn open_read_only(path: &Path) -> rusqlite::Result<Connection> {
    // Without `SQLITE_OPEN_CREATE` a missing file is an error rather than a
    // new empty database.
    Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
}

/// Turns free text into an FTS5 query that can't be a syntax error: each
/// word becomes a quoted string, OR-ed together so a long query still
/// matches, with bm25 ranking the issues that match more words first.
/// Numbers at or above `later_from` are dropped: in an earlier issue they
/// can only be references added by a later edit (see the module docs).
pub fn fts_query(text: &str, later_from: u64) -> Option<String> {
    let terms: Vec<String> = text
        .split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|t| !t.is_empty())
        .filter(|t| !t.parse::<u64>().is_ok_and(|n| n >= later_from))
        .take(MAX_QUERY_TERMS)
        .map(|t| format!("\"{}\"", t.replace('"', "\"\"")))
        .collect();
    (!terms.is_empty()).then(|| terms.join(" OR "))
}

/// References to issues and pull requests: a GitHub URL, a `issues/N` or
/// `pull/N` path, or `#N` with an optional `owner/repo` in front.
static ISSUE_REF: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)(?:https?://)?(?:www\.)?github\.com/(?P<url_repo>[\w.-]+/[\w.-]+)/(?:issues|pull)/(?P<url_n>\d+)(?:#[\w-]+)?|\b(?:issues|pull)/(?P<path_n>\d+)|(?P<short_repo>\b[\w.-]+/[\w.-]+)?#(?P<short_n>\d+)\b",
    )
    .expect("a valid regex")
});
/// An FTS snippet that starts in the middle of a body, right at a number: the
/// `issues/` or `#` before it was cut off.
static SNIPPET_LEADING_NUMBER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(?P<lead>\s*\.\.\.\s*)(?P<n>\d+)\b").expect("a valid regex"));

/// `text` with every reference to this repo's issues or pull requests
/// numbered `later_from` or above replaced by [`LATER_REF`]. References to
/// other repos are kept: their numbers are a different sequence.
pub fn redact_later_refs<'a>(text: &'a str, repo: &str, later_from: u64) -> Cow<'a, str> {
    let is_later = |n: &str| n.parse::<u64>().is_ok_and(|n| n >= later_from);
    let same_repo =
        |r: Option<regex::Match>| r.is_none_or(|r| r.as_str().eq_ignore_ascii_case(repo));
    ISSUE_REF.replace_all(text, |c: &Captures| {
        let (n, repo_match) = if let Some(n) = c.name("url_n") {
            (n, c.name("url_repo"))
        } else if let Some(n) = c.name("path_n") {
            (n, None)
        } else {
            (
                c.name("short_n").expect("one branch matched"),
                c.name("short_repo"),
            )
        };
        if is_later(n.as_str()) && same_repo(repo_match) {
            LATER_REF.to_owned()
        } else {
            c[0].to_owned()
        }
    })
}

fn redact_snippet(snippet: &str, repo: &str, later_from: u64) -> String {
    let text = redact_later_refs(snippet, repo, later_from);
    SNIPPET_LEADING_NUMBER
        .replace(&text, |c: &Captures| {
            if c["n"].parse::<u64>().is_ok_and(|n| n >= later_from) {
                format!("{}{LATER_REF}", &c["lead"])
            } else {
                c[0].to_owned()
            }
        })
        .into_owned()
}

fn search_issues(
    conn: &Connection,
    ctx: &ToolContext,
    query: &str,
    limit: u32,
) -> Result<Value, String> {
    let (repo, later_from) = (ctx.repo.name.as_str(), ctx.issue_number);
    let Some(fts) = fts_query(query, later_from) else {
        return Err("the query has no words to search for".into());
    };
    let mut stmt = conn
        .prepare(
            "SELECT i.number, i.title, i.created_at,
                    snippet(issues_fts, 1, '', '', ' ... ', 24)
             FROM issues_fts JOIN issues i ON i.number = issues_fts.rowid
             WHERE issues_fts MATCH ?1 AND i.created_at < ?2
             ORDER BY bm25(issues_fts, 3.0, 1.0)
             LIMIT ?3",
        )
        .map_err(db_error)?;
    let rows = stmt
        .query_map(params![fts, ctx.repo.as_of, limit], |row| {
            Ok(json!({
                "number": row.get::<_, i64>(0)?,
                "title": redact_later_refs(&row.get::<_, String>(1)?, repo, later_from),
                "created_at": row.get::<_, String>(2)?,
                "snippet": redact_snippet(&row.get::<_, String>(3)?, repo, later_from),
            }))
        })
        .map_err(db_error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(db_error)?;
    Ok(json!(rows))
}

fn get_issue(conn: &Connection, ctx: &ToolContext, number: u64) -> Result<Value, String> {
    let (repo, later_from) = (ctx.repo.name.as_str(), ctx.issue_number);
    let issue = conn
        .query_row(
            "SELECT number, title, body, created_at FROM issues
             WHERE number = ?1 AND created_at < ?2",
            params![number as i64, ctx.repo.as_of],
            |row| {
                let body = row.get::<_, String>(2)?;
                Ok(json!({
                    "number": row.get::<_, i64>(0)?,
                    "title": redact_later_refs(&row.get::<_, String>(1)?, repo, later_from),
                    "body": truncate(&redact_later_refs(&body, repo, later_from), MAX_BODY_CHARS),
                    "created_at": row.get::<_, String>(3)?,
                }))
            },
        )
        .optional()
        .map_err(db_error)?;
    issue.ok_or_else(|| format!("no issue #{number} filed before this one"))
}

fn get_issue_templates(conn: &Connection) -> Result<Value, String> {
    let mut stmt = conn
        .prepare("SELECT name, body FROM templates ORDER BY name")
        .map_err(db_error)?;
    let rows = stmt
        .query_map([], |row| {
            Ok(json!({
                "name": row.get::<_, String>(0)?,
                "body": truncate(&row.get::<_, String>(1)?, MAX_BODY_CHARS),
            }))
        })
        .map_err(db_error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(db_error)?;
    Ok(json!(rows))
}

fn get_releases(conn: &Connection, as_of: &str) -> Result<Value, String> {
    let mut stmt = conn
        .prepare(
            "SELECT tag, published_at, prerelease FROM releases
             WHERE published_at < ?1 ORDER BY published_at DESC LIMIT ?2",
        )
        .map_err(db_error)?;
    let rows = stmt
        .query_map(params![as_of, MAX_RELEASES], |row| {
            Ok(json!({
                "tag": row.get::<_, String>(0)?,
                "published_at": row.get::<_, String>(1)?,
                "prerelease": row.get::<_, i64>(2)? != 0,
            }))
        })
        .map_err(db_error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(db_error)?;
    Ok(json!(rows))
}

fn db_error(e: rusqlite::Error) -> String {
    format!("corpus query failed: {e}")
}

fn truncate(text: &str, max_chars: usize) -> String {
    match text.char_indices().nth(max_chars) {
        Some((i, _)) => format!("{}\n[... truncated]", &text[..i]),
        None => text.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use triage_core::Label;

    fn repo(as_of: &str) -> Repo {
        Repo {
            name: "kubernetes/kubernetes".into(),
            labels: vec![Label {
                name: "sig/node".into(),
                description: Some("Node".into()),
            }],
            latest_release: Some("v1.33.2".into()),
            as_of: as_of.into(),
        }
    }

    fn corpus(dir: &Path) {
        let conn = Connection::open(dir.join("kubernetes__kubernetes.sqlite")).unwrap();
        conn.execute_batch(
            r#"
            CREATE TABLE issues (number INTEGER PRIMARY KEY, title TEXT NOT NULL, body TEXT NOT NULL, created_at TEXT NOT NULL, state TEXT NOT NULL);
            CREATE VIRTUAL TABLE issues_fts USING fts5(title, body, content='issues', content_rowid='number');
            CREATE TABLE releases (tag TEXT PRIMARY KEY, published_at TEXT NOT NULL, prerelease INTEGER NOT NULL);
            CREATE TABLE templates (name TEXT PRIMARY KEY, body TEXT NOT NULL);
            INSERT INTO issues VALUES
              (100, 'kubelet crashes on startup', 'panic: nil pointer in kubelet', '2025-01-01T00:00:00Z', 'closed'),
              (200, 'kubelet crash after upgrade', 'kubelet panics again', '2025-06-01T00:00:00Z', 'open'),
              (300, 'kubectl apply is slow', 'apply takes minutes', '2025-02-01T00:00:00Z', 'open');
            INSERT INTO issues_fts(issues_fts) VALUES('rebuild');
            INSERT INTO releases VALUES ('v1.32.0', '2024-12-11T00:00:00Z', 0), ('v1.34.0', '2025-08-27T00:00:00Z', 0);
            INSERT INTO templates VALUES ('bug-report.yaml', 'What happened?');
            "#,
        )
        .unwrap();
    }

    fn numbers(v: &Value) -> Vec<i64> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|r| r["number"].as_i64().unwrap())
            .collect()
    }

    #[test]
    fn search_respects_as_of() {
        let dir = tempfile::tempdir().unwrap();
        corpus(dir.path());
        let ctx = ToolContext::with_corpus_dir(repo("2025-03-01T00:00:00Z"), 1000, dir.path());

        let found = dispatch(&ctx, SEARCH_ISSUES, r#"{"query": "kubelet crash"}"#).unwrap();
        assert_eq!(numbers(&found), vec![100]);
        assert!(found[0].get("state").is_none());

        let later = ToolContext::with_corpus_dir(repo("2025-07-01T00:00:00Z"), 1000, dir.path());
        let found = dispatch(&later, SEARCH_ISSUES, r#"{"query": "kubelet crash"}"#).unwrap();
        assert_eq!(numbers(&found).len(), 2);
    }

    #[test]
    fn get_issue_respects_as_of() {
        let dir = tempfile::tempdir().unwrap();
        corpus(dir.path());
        let ctx = ToolContext::with_corpus_dir(repo("2025-03-01T00:00:00Z"), 1000, dir.path());

        let issue = dispatch(&ctx, GET_ISSUE, r#"{"number": 100}"#).unwrap();
        assert_eq!(issue["title"], "kubelet crashes on startup");
        let err = dispatch(&ctx, GET_ISSUE, r#"{"number": 200}"#).unwrap_err();
        assert!(err.contains("no issue #200"), "{err}");
    }

    #[test]
    fn fts_syntax_cannot_break_the_query() {
        let dir = tempfile::tempdir().unwrap();
        corpus(dir.path());
        let ctx = ToolContext::with_corpus_dir(repo("2025-03-01T00:00:00Z"), 1000, dir.path());

        for (query, expected) in [
            (r#"kubelet" AND (crash"#, vec![100]),
            ("kubelet NOT* NEAR(", vec![100]),
            ("title:kubelet ^crash", vec![100]),
            ("-kubelet", vec![100]),
            ("\"unterminated", vec![]),
        ] {
            let args = json!({ "query": query }).to_string();
            let found =
                dispatch(&ctx, SEARCH_ISSUES, &args).unwrap_or_else(|e| panic!("{query:?}: {e}"));
            assert_eq!(numbers(&found), expected, "{query:?}");
        }
        assert!(dispatch(&ctx, SEARCH_ISSUES, r#"{"query": "()*"}"#).is_err());
        assert_eq!(
            fts_query(r#"a"b c"#, 1000).as_deref(),
            Some(r#""a" OR "b" OR "c""#)
        );
        // Numbers from the triaged issue's on are dropped.
        assert_eq!(
            fts_query("kubelet 999 1000 123456", 1000).as_deref(),
            Some(r#""kubelet" OR "999""#)
        );
        assert_eq!(fts_query("1000", 1000), None);
    }

    #[test]
    fn redacts_references_to_later_issues() {
        let dir = tempfile::tempdir().unwrap();
        corpus(dir.path());
        let conn = Connection::open(dir.path().join("kubernetes__kubernetes.sqlite")).unwrap();
        // #50 was edited later to list the duplicates filed after it.
        conn.execute_batch(
            r#"
            INSERT INTO issues VALUES (50, 'scheduler hangs (see #1200)',
              'The scheduler hangs. Duplicates: https://github.com/kubernetes/kubernetes/issues/1042, kubernetes/kubernetes#1100, #1300, /pull/1400. Related: #40, https://github.com/kubernetes/kubernetes/issues/30, kubernetes/enhancements#5000',
              '2024-06-01T00:00:00Z', 'closed');
            INSERT INTO issues_fts(issues_fts) VALUES('rebuild');
            "#,
        )
        .unwrap();
        let ctx = ToolContext::with_corpus_dir(repo("2025-03-01T00:00:00Z"), 1042, dir.path());

        let issue = dispatch(&ctx, GET_ISSUE, r#"{"number": 50}"#).unwrap();
        let body = issue["body"].as_str().unwrap();
        for later in ["1042", "1100", "1300", "1400"] {
            assert!(!body.contains(later), "{later} in {body}");
        }
        assert_eq!(body.matches(LATER_REF).count(), 4, "{body}");
        // Earlier issues and other repos stay.
        assert!(
            body.contains("#40") && body.contains("/issues/30"),
            "{body}"
        );
        assert!(body.contains("kubernetes/enhancements#5000"), "{body}");
        assert_eq!(issue["title"], format!("scheduler hangs (see {LATER_REF})"));

        // Searching for its own number doesn't find the issue edited to point
        // at it.
        let err = dispatch(&ctx, SEARCH_ISSUES, r#"{"query": "1042"}"#).unwrap_err();
        assert!(err.contains("no words"), "{err}");
        let found = dispatch(&ctx, SEARCH_ISSUES, r#"{"query": "scheduler duplicates"}"#).unwrap();
        assert_eq!(numbers(&found), vec![50]);
        let shown = found.to_string();
        assert!(
            !shown.contains("1042") && !shown.contains("1200"),
            "{shown}"
        );
    }

    #[test]
    fn redacts_a_snippet_cut_right_before_a_later_number() {
        assert_eq!(
            redact_snippet(" ... 1042, and #7 ...", "kubernetes/kubernetes", 1000),
            format!(" ... {LATER_REF}, and #7 ...")
        );
        assert_eq!(
            redact_later_refs(
                "GitHub.com/Kubernetes/Kubernetes/pull/1001#issuecomment-1",
                "kubernetes/kubernetes",
                1000
            ),
            LATER_REF
        );
    }

    #[test]
    fn releases_before_as_of() {
        let dir = tempfile::tempdir().unwrap();
        corpus(dir.path());
        let ctx = ToolContext::with_corpus_dir(repo("2025-03-01T00:00:00Z"), 1000, dir.path());
        let releases = dispatch(&ctx, GET_RELEASES, "").unwrap();
        assert_eq!(releases.as_array().unwrap().len(), 1);
        assert_eq!(releases[0]["tag"], "v1.32.0");
        let templates = dispatch(&ctx, GET_ISSUE_TEMPLATES, "{}").unwrap();
        assert_eq!(templates[0]["name"], "bug-report.yaml");
    }

    #[test]
    fn missing_corpus_is_a_message() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolContext::with_corpus_dir(repo("2025-03-01T00:00:00Z"), 1000, dir.path());
        let err = dispatch(&ctx, SEARCH_ISSUES, r#"{"query": "x"}"#).unwrap_err();
        assert!(err.contains("not available"), "{err}");
        // Labels come from the request, not the corpus.
        assert!(dispatch(&ctx, LIST_LABELS, "{}").is_ok());
        assert!(
            dispatch(&ctx, "nope", "{}")
                .unwrap_err()
                .contains("unknown tool")
        );
    }

    #[test]
    fn definitions_have_object_schemas() {
        for d in definitions() {
            assert_eq!(d.parameters["type"], "object", "{}", d.name);
        }
    }
}
