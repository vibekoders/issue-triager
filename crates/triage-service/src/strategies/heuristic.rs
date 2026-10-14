//! `Heuristic`: regexes and parsing only, the no-AI baseline. It reads the
//! Kubernetes issue templates' sections, code blocks, stack traces, version
//! strings and keywords.
//!
//! The confidences are fixed per rule, not calibrated: a template match is
//! worth more than a keyword. Duplicates and a reply need search or
//! judgement, so they are left out.
//!
//! Version 3: a field the text gives no signal for is left out, not guessed
//! (the plan's contract). That is kind when no template, title or keyword
//! rule fires, area when no test tag, keyword or mined token does, and
//! severity always: nothing in an issue's text predicted its priority. The
//! v2 base rates are still there behind `base_rates: true` in the config, as
//! a separately labelled prior-only baseline.
//!
//! Version 2 (experiment 1), tuned on dev-split issues only (dev minus
//! dev-200, with dev-200 held out to check; never the test split):
//! - kind: flake markers in the title ("[Flaking Test]", which v1 missed);
//!   "Failure cluster [...]" titles are failing tests; title prefixes for
//!   features ("Add ", "KEP-", "[FG:...]") and cleanups ("Remove ", "Add
//!   unit test") in issues without a template. A template the author picked
//!   still wins over the title.
//! - area: e2e test tags (`[sig-storage] ...`) decide outright; otherwise the
//!   hand-written [`SIG_KEYWORDS`] plus a token table mined from dev-split
//!   issues ([`super::sig_tokens`], `scripts/mine_sig_keywords.py`); with no
//!   signal at all, the most common SIG (only with `base_rates` since v3).
//! - severity: the most common priority for the (predicted) kind in dev-split
//!   issues, with that share as the confidence: a base rate, not a judgement
//!   (only with `base_rates` since v3).
//! - is_regression: only an explicit version in the title ("... in 1.31"),
//!   an upgrade from one version to another, or the issue calling itself a
//!   regression; v1's loose phrases ("after upgrading") were wrong 9 times
//!   out of 10.
//! - enough_info: false only when the report has hardly anything to go on.
//!   In dev-split bugs 85% have enough information, and v1's stricter
//!   checklist was wrong on most of the reports it called insufficient.

use std::sync::LazyLock;

use std::collections::{BTreeMap, BTreeSet};

use regex::Regex;
use restate_sdk::prelude::*;
use tracing::{Instrument, info};
use triage_core::config::HeuristicConfig;
use triage_core::{
    Field, Issue, Kind, Repo, Reproducibility, Scope, Severity, Triage, TriageRequest,
    TriageResponse, Usage,
};

use super::sig_tokens::SIG_TOKENS;
use super::{parse_config, record_run, response};
use crate::trace::handler_span;

pub const VERSION: &str = "3";
const NAME: &str = "Heuristic";

pub struct HeuristicStrategy;

/// Timeouts: see "Restate timeouts" in the [`super`] module docs.
#[restate_sdk::workflow(name = "Heuristic", inactivity_timeout = "5m", abort_timeout = "1h")]
impl HeuristicStrategy {
    #[handler]
    async fn run(
        &self,
        ctx: WorkflowContext<'_>,
        Json(request): Json<TriageRequest>,
    ) -> Result<Json<TriageResponse>, HandlerError> {
        let span = handler_span(NAME, ctx.key(), ctx.headers());
        async move {
            let (request, config_hash) = parse_config::<HeuristicConfig>(request)?;
            record_run(&tracing::Span::current(), VERSION, &config_hash);
            // Pure and deterministic, so it needs no `ctx.run`: a replay
            // computes the same answer.
            let triage = triage(&request.issue, &request.repo, &request.config);
            info!(
                kind = triage.kind.as_ref().map(|k| k.value.as_str()),
                area = triage.area.as_ref().map(|a| a.value.as_str()),
                "heuristic triage done"
            );
            Ok(Json(response(
                NAME,
                VERSION,
                config_hash,
                triage,
                Usage::default(),
            )))
        }
        .instrument(span)
        .await
    }
}

/// Keywords that point at each SIG, for the Kubernetes repo. Also used as
/// scope hints in prompts, since the repo's own label descriptions only say
/// "Categorizes an issue or PR as relevant to SIG X". Hand-written; the
/// heuristic adds the mined [`SIG_TOKENS`] to them, but the prompts don't.
pub const SIG_KEYWORDS: &[(&str, &[&str])] = &[
    (
        "sig/node",
        &[
            "kubelet",
            "pod",
            "container",
            "cri",
            "containerd",
            "cri-o",
            "crio",
            "cgroup",
            "cgroupv2",
            "oom",
            "oomkilled",
            "eviction",
            "evicted",
            "probe",
            "liveness",
            "readiness",
            "static pod",
            "runtimeclass",
            "swap",
            "device plugin",
            "node",
            "image pull",
            "sandbox",
            "sidecar",
            "init container",
            "ephemeral container",
        ],
    ),
    ("sig/cli", &["kubectl", "kustomize", "kubeconfig"]),
    (
        "sig/scheduling",
        &[
            "scheduler",
            "kube-scheduler",
            "scheduling",
            "affinity",
            "anti-affinity",
            "taint",
            "toleration",
            "preemption",
            "priorityclass",
            "topology spread",
            "nodeselector",
            "unschedulable",
        ],
    ),
    (
        "sig/api-machinery",
        &[
            "apiserver",
            "kube-apiserver",
            "api server",
            "api-server",
            "etcd",
            "crd",
            "customresourcedefinition",
            "custom resource",
            "webhook",
            "admission",
            "informer",
            "client-go",
            "garbage collector",
            "finalizer",
            "server-side apply",
            "managedfields",
            "openapi",
            "apiservice",
            "watch cache",
            "resourceversion",
            "apimachinery",
            "validatingadmissionpolicy",
        ],
    ),
    (
        "sig/network",
        &[
            "network",
            "networking",
            "service",
            "ingress",
            "dns",
            "coredns",
            "kube-proxy",
            "iptables",
            "ipvs",
            "nftables",
            "endpoint",
            "endpointslice",
            "loadbalancer",
            "load balancer",
            "nodeport",
            "clusterip",
            "cni",
            "networkpolicy",
            "network policy",
            "ipv6",
            "dual-stack",
            "gateway api",
            "conntrack",
        ],
    ),
    (
        "sig/storage",
        &[
            "volume",
            "pv",
            "pvc",
            "persistentvolume",
            "persistentvolumeclaim",
            "csi",
            "storage",
            "storageclass",
            "mount",
            "unmount",
            "attach",
            "detach",
            "nfs",
            "volumesnapshot",
            "snapshot",
            "emptydir",
            "hostpath",
            "fsgroup",
            "subpath",
        ],
    ),
    (
        "sig/auth",
        &[
            "rbac",
            "authentication",
            "authorization",
            "authn",
            "authz",
            "serviceaccount",
            "service account",
            "token",
            "certificate",
            "csr",
            "oidc",
            "clusterrole",
            "rolebinding",
            "kms",
            "impersonation",
            "pod security",
            "podsecurity",
        ],
    ),
    (
        "sig/apps",
        &[
            "deployment",
            "statefulset",
            "daemonset",
            "replicaset",
            "job",
            "cronjob",
            "rollout",
            "rolling update",
            "poddisruptionbudget",
            "pdb",
            "replicationcontroller",
        ],
    ),
    (
        "sig/autoscaling",
        &[
            "hpa",
            "horizontalpodautoscaler",
            "horizontal pod autoscaler",
            "autoscaler",
            "autoscaling",
            "vpa",
        ],
    ),
    (
        "sig/cloud-provider",
        &[
            "cloud provider",
            "cloud-provider",
            "cloud-controller-manager",
            "aws",
            "azure",
            "gce",
            "gcp",
            "openstack",
            "vsphere",
        ],
    ),
    (
        "sig/instrumentation",
        &[
            "metrics",
            "metric",
            "prometheus",
            "klog",
            "logging",
            "structured logging",
            "tracing",
            "opentelemetry",
            "metrics-server",
        ],
    ),
    (
        "sig/testing",
        &[
            "e2e",
            "test-infra",
            "prow",
            "flaky",
            "flake",
            "ginkgo",
            "integration test",
            "unit test",
            "testgrid",
        ],
    ),
    (
        "sig/release",
        &[
            "release notes",
            "changelog",
            "release branch",
            "cherry-pick",
            "cherry pick",
        ],
    ),
    ("sig/windows", &["windows", "hostprocess", "hns"]),
    (
        "sig/cluster-lifecycle",
        &["kubeadm", "kops", "cluster-api", "kubespray"],
    ),
    ("sig/k8s-infra", &["registry.k8s.io", "k8s.gcr.io"]),
    (
        "sig/scalability",
        &[
            "scalability",
            "kubemark",
            "throughput",
            "performance regression",
        ],
    ),
    ("sig/architecture", &["conformance", "deprecation policy"]),
    ("sig/docs", &["documentation", "docs"]),
    ("sig/security", &["cve", "vulnerability", "govulncheck"]),
];

/// The whole heuristic triage.
pub fn triage(issue: &Issue, repo: &Repo, config: &HeuristicConfig) -> Triage {
    let doc = Doc::parse(&issue.title, &issue.body);
    let base_rates = config.base_rates;
    let kind = kind(&doc).or_else(|| base_rates.then(kind_base_rate));
    let kind_value = kind.as_ref().map(|k| k.value);
    // Like the other strategies: a checklist for bug-like or unknown kinds.
    let reproducibility = match kind_value {
        Some(Kind::Bug | Kind::FailingTest | Kind::Flake) | None => Some(reproducibility(&doc)),
        _ => None,
    };
    let area = area(&doc, repo).or_else(|| base_rates.then(|| area_base_rate(repo)).flatten());
    let severity = base_rates
        .then(|| kind_value.and_then(severity_base_rate))
        .flatten();
    Triage {
        area,
        severity,
        is_regression: Some(is_regression(&doc)),
        security_sensitive: Some(security_sensitive(&doc)),
        quality: Some(quality(&doc, kind_value)),
        outdated_version: outdated_version(&doc, repo),
        scope: Some(scope(&doc)),
        reproducibility,
        kind,
        duplicates: None,
        suggested_reply: None,
    }
}

/// An issue split into its template sections.
struct Doc {
    title: String,
    /// Lowercased title and body.
    text: String,
    body_len: usize,
    /// `(heading, content)`, headings lowercased. Content before the first
    /// heading has an empty heading.
    sections: Vec<(String, String)>,
    code_blocks: usize,
    /// Words of the title and body, as `scripts/mine_sig_keywords.py` splits
    /// them (see [`tokens`]).
    tokens: BTreeSet<String>,
    title_tokens: BTreeSet<String>,
}

static HTML_COMMENT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?s)<!--.*?-->").unwrap());
static BOLD_HEADING: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\*\*([^*]{3,80})\*\*\s*:?\s*$").unwrap());

impl Doc {
    fn parse(title: &str, body: &str) -> Doc {
        let body = HTML_COMMENT.replace_all(body, "");
        let mut sections = vec![(String::new(), String::new())];
        let mut in_fence = false;
        // Only code blocks with something in them count: the templates come
        // with empty ones to paste into.
        let mut code_blocks = 0;
        let mut fenced = String::new();
        for line in body.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
                if in_fence && meaningful(&fenced) {
                    code_blocks += 1;
                }
                fenced.clear();
                in_fence = !in_fence;
            } else if in_fence {
                fenced.push_str(line);
                fenced.push('\n');
            } else if let Some(heading) = heading(trimmed) {
                sections.push((heading.to_lowercase(), String::new()));
                continue;
            }
            let content = &mut sections.last_mut().expect("never empty").1;
            content.push_str(line);
            content.push('\n');
        }
        let text = format!("{title}\n{body}").to_lowercase();
        Doc {
            title: title.to_owned(),
            body_len: body.trim().len(),
            sections,
            code_blocks,
            tokens: tokens(&text),
            title_tokens: tokens(&title.to_lowercase()),
            text,
        }
    }

    /// The first section whose heading contains one of `needles`, if its
    /// content is more than template boilerplate.
    fn filled(&self, needles: &[&str]) -> Option<&str> {
        self.sections
            .iter()
            .find(|(h, _)| needles.iter().any(|n| h.contains(n)))
            .map(|(_, c)| c.as_str())
            .filter(|c| meaningful(c))
    }

    fn has_heading(&self, needles: &[&str]) -> bool {
        self.sections
            .iter()
            .any(|(h, _)| needles.iter().any(|n| h.contains(n)))
    }
}

static TOKEN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[a-z][a-z0-9]*(?:[-_.][a-z0-9]+)*").unwrap());
static MENTION: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"@[a-z0-9][a-z0-9-]*").unwrap());

/// Splits lowercased text (HTML comments already removed) into words the
/// way `scripts/mine_sig_keywords.py` does: @mentions dropped, 3 to 40
/// characters, fewer than three digits.
fn tokens(text: &str) -> BTreeSet<String> {
    let text = MENTION.replace_all(text, " ");
    TOKEN
        .find_iter(&text)
        .map(|m| m.as_str())
        .filter(|t| (3..=40).contains(&t.len()))
        .filter(|t| t.chars().filter(char::is_ascii_digit).count() < 3)
        .map(str::to_owned)
        .collect()
}

fn heading(line: &str) -> Option<&str> {
    let hashes = line.chars().take_while(|&c| c == '#').count();
    if (1..=6).contains(&hashes) && line[hashes..].starts_with(' ') {
        return Some(line[hashes..].trim().trim_end_matches(':'));
    }
    BOLD_HEADING
        .captures(line)
        .and_then(|c| c.get(1))
        .map(|m| m.as_str().trim().trim_end_matches(':'))
}

/// Whether a section has content beyond the template's placeholders.
fn meaningful(content: &str) -> bool {
    const PLACEHOLDERS: &[&str] = &[
        "_no response_",
        "<details>",
        "</details>",
        "$ kubectl version",
        "# paste output here",
        "n/a",
        "na",
        "none",
        "-",
        "no",
    ];
    let rest: usize = content
        .lines()
        .map(str::trim)
        .filter(|l| !l.starts_with("```") && !l.starts_with("~~~"))
        .filter(|l| !PLACEHOLDERS.contains(&l.to_lowercase().as_str()))
        .map(str::len)
        .sum();
    rest >= 3
}

fn contains_any(text: &str, needles: &[&'static str]) -> Option<&'static str> {
    needles.iter().copied().find(|n| text.contains(n))
}

const BUG_HEADINGS: &[&str] = &["what happened", "what did you expect to happen"];
const FEATURE_HEADINGS: &[&str] = &["what would you like to be added", "why is this needed"];
const FAILING_HEADINGS: &[&str] = &["which jobs are failing", "which tests are failing"];
const FLAKE_HEADINGS: &[&str] = &["which jobs are flaking", "which tests are flaking"];

/// Title markers of a flaky test. "flak" covers flaky, flaking and flake.
const FLAKE_TITLE: &[&str] = &["flak"];
const FAILING_TITLE: &[&str] = &[
    "[failing test]",
    "[failing-test]",
    "failing test",
    "test failure",
    "failure cluster",
];
/// How issues without a template (mostly filed by members) start their title
/// when they ask for something new. Checked in dev-split issues.
const FEATURE_TITLE_PREFIXES: &[&str] = &[
    "kep-",
    "kep ",
    "[kep",
    "[umbrella]",
    "[fg:",
    "was:",
    "add ",
    "support ",
    "implement ",
    "introduce ",
    "expose ",
    "allow ",
    "enable ",
    "proposal",
    "idea:",
    "feature request",
    "[feature]",
    "promote ",
];
/// Checked before the feature prefixes: "Add e2e test for ..." is a cleanup.
const CLEANUP_TITLE_PREFIXES: &[&str] = &[
    "add e2e test",
    "add unit test",
    "add test",
    "add integration test",
    "add additional test",
    "add more test",
    "remove ",
    "replace ",
    "migrate ",
    "deprecate ",
    "drop ",
    "rename ",
    "refactor",
    "cleanup",
    "clean up",
    "clean-up",
    "bump ",
];

/// The kind the text points at; `None` when nothing does.
fn kind(doc: &Doc) -> Option<Field<Kind>> {
    let title = doc.title.to_lowercase();
    let title = title.trim();
    let template = |kind, name: &str| {
        Field::new(kind)
            .with_confidence(0.85)
            .with_evidence(format!("{name} template"))
    };
    // Test triage dashboard reports; nearly always a failing test.
    if title.starts_with("failure cluster") {
        return Some(keyword(Kind::FailingTest, 0.8, "failure cluster"));
    }
    if doc.has_heading(FLAKE_HEADINGS) {
        return Some(template(Kind::Flake, "flaking test"));
    }
    if doc.has_heading(FAILING_HEADINGS) {
        return Some(template(Kind::FailingTest, "failing test"));
    }
    if let Some(k) = contains_any(title, FLAKE_TITLE) {
        return Some(keyword(Kind::Flake, 0.75, k));
    }
    if let Some(k) = contains_any(title, FAILING_TITLE) {
        return Some(keyword(Kind::FailingTest, 0.7, k));
    }
    if doc.has_heading(FEATURE_HEADINGS) {
        return Some(
            Field::new(Kind::Feature)
                .with_confidence(0.75)
                .with_evidence("feature request template"),
        );
    }
    if doc.has_heading(BUG_HEADINGS) {
        // The bug template is also the default one, so people file support
        // questions and docs fixes with it: worth less than the others.
        return Some(
            Field::new(Kind::Bug)
                .with_confidence(0.7)
                .with_evidence("bug report template"),
        );
    }
    if let Some(k) = CLEANUP_TITLE_PREFIXES
        .iter()
        .find(|p| title.starts_with(*p))
    {
        return Some(keyword(Kind::Cleanup, 0.55, k));
    }
    if let Some(k) = FEATURE_TITLE_PREFIXES
        .iter()
        .find(|p| title.starts_with(*p))
    {
        return Some(keyword(Kind::Feature, 0.55, k));
    }
    let text = &doc.text;
    if let Some(k) = contains_any(
        text,
        &["typo in the doc", "documentation", "in the docs", "docs:"],
    ) {
        return Some(keyword(Kind::Docs, 0.45, k));
    }
    if let Some(k) = contains_any(
        title,
        &[
            "cleanup",
            "clean up",
            "refactor",
            "remove deprecated",
            "bump ",
            "tech debt",
        ],
    ) {
        return Some(keyword(Kind::Cleanup, 0.5, k));
    }
    if let Some(k) = contains_any(
        title,
        &[
            "feature request",
            "support for",
            "add support",
            "proposal",
            "enhancement",
            "kep",
            "allow ",
            "should support",
        ],
    ) {
        return Some(keyword(Kind::Feature, 0.5, k));
    }
    if let Some(k) = contains_any(
        title,
        &[
            "how to",
            "how do i",
            "how can i",
            "is it possible",
            "question",
        ],
    ) {
        return Some(keyword(Kind::Question, 0.45, k));
    }
    contains_any(
        text,
        &[
            "panic",
            "crash",
            "error",
            "fail",
            "broken",
            "bug",
            "regression",
            "doesn't work",
            "does not work",
            "not working",
            "unexpected",
            "wrong",
        ],
    )
    .map(|k| keyword(Kind::Bug, 0.5, k))
}

/// With `base_rates`, when nothing points at a kind: most Kubernetes issues
/// are bugs.
fn kind_base_rate() -> Field<Kind> {
    Field::new(Kind::Bug)
        .with_confidence(0.3)
        .with_evidence("no signal; base rate: most issues are bugs")
}

fn keyword(kind: Kind, confidence: f64, keyword: &str) -> Field<Kind> {
    Field::new(kind)
        .with_confidence(confidence)
        .with_evidence(format!("keyword {:?}", keyword.trim()))
}

/// One regex per SIG, matching any of its keywords as whole words (with an
/// optional plural `s`).
static SIG_PATTERNS: LazyLock<Vec<(&'static str, Regex)>> = LazyLock::new(|| {
    SIG_KEYWORDS
        .iter()
        .map(|(sig, words)| {
            let alternatives: Vec<String> = words.iter().map(|w| regex::escape(w)).collect();
            let pattern = format!(r"\b(?:{})s?\b", alternatives.join("|"));
            (*sig, Regex::new(&pattern).unwrap())
        })
        .collect()
});

/// e2e test names carry their SIG: `[sig-storage] CSI Volumes ...`.
static TEST_TAG: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\[sig-([a-z-]+)\]").unwrap());

/// The most common SIG label in dev-split issues (793 of 2185 tuning issues
/// with a sig label): the `base_rates` guess when nothing in the issue points
/// anywhere.
const AREA_FALLBACK: &str = "sig/node";
/// How much a hand-written keyword hit counts next to the mined tokens'
/// log-odds weights. Picked on the dev tuning issues.
const HAND_KEYWORD_WEIGHT: f64 = 0.5;

/// The repo's label for a SIG name as test tags write it, comparing only
/// letters and digits (`[sig-apimachinery]` is `sig/api-machinery`).
fn sig_label<'a>(repo: &'a Repo, name: &str) -> Option<&'a str> {
    let squash = |s: &str| -> String { s.chars().filter(char::is_ascii_alphanumeric).collect() };
    let wanted = squash(name);
    repo.labels
        .iter()
        .filter_map(|l| l.name.strip_prefix("sig/").map(|n| (l, n)))
        .find(|(_, n)| squash(n) == wanted)
        .map(|(l, _)| l.name.as_str())
}

fn area(doc: &Doc, repo: &Repo) -> Option<Field<String>> {
    // 1. An e2e test tag names the SIG that owns the test.
    let mut tags: BTreeMap<&str, usize> = BTreeMap::new();
    for c in TEST_TAG.captures_iter(&doc.text) {
        if let Some(label) = sig_label(repo, &c[1]) {
            *tags.entry(label).or_default() += 1;
        }
    }
    if let Some((label, _)) = tags.iter().max_by_key(|(_, n)| **n) {
        let tag = label.replacen('/', "-", 1);
        return Some(
            Field::new((*label).to_owned())
                .with_confidence(0.85)
                .with_evidence(format!("test tag [{tag}]")),
        );
    }

    // 2. Keywords: the hand-written ones (title matches count three times;
    //    each keyword counts at most three times, so a pasted log that
    //    repeats one word doesn't decide alone) plus the mined tokens
    //    (present or not; three times in the title).
    let in_repo = |sig: &str| repo.labels.iter().any(|l| l.name == sig);
    let title = doc.title.to_lowercase();
    let mut scores: BTreeMap<&str, (f64, Vec<String>)> = BTreeMap::new();
    for (sig, re) in SIG_PATTERNS.iter().filter(|(sig, _)| in_repo(sig)) {
        let mut counts: Vec<(String, usize)> = Vec::new();
        for m in re.find_iter(&doc.text) {
            let word = m.as_str().trim_end_matches('s').to_owned();
            match counts.iter_mut().find(|(w, _)| *w == word) {
                Some((_, n)) => *n += 1,
                None => counts.push((word, 1)),
            }
        }
        let body: usize = counts.iter().map(|(_, n)| (*n).min(3)).sum();
        let in_title = re.find_iter(&title).count();
        let score = HAND_KEYWORD_WEIGHT * (body + 2 * in_title) as f64;
        if score > 0.0 {
            let entry = scores.entry(sig).or_default();
            entry.0 += score;
            entry.1.extend(counts.into_iter().map(|(w, _)| w));
        }
    }
    for (sig, table) in SIG_TOKENS.iter().filter(|(sig, _)| in_repo(sig)) {
        for (token, weight) in table.iter().filter(|(t, _)| doc.tokens.contains(*t)) {
            let factor = if doc.title_tokens.contains(*token) {
                3.0
            } else {
                1.0
            };
            let entry = scores.entry(sig).or_default();
            entry.0 += weight * factor;
            if !entry.1.iter().any(|w| w == token) {
                entry.1.push((*token).to_owned());
            }
        }
    }
    let total: f64 = scores.values().map(|(s, _)| s).sum();
    let best = scores
        .into_iter()
        .filter(|(_, (score, _))| *score > 0.0)
        // Highest score; ties go to the alphabetically first SIG.
        .max_by(|a, b| a.1.0.total_cmp(&b.1.0).then_with(|| b.0.cmp(a.0)));
    if let Some((sig, (score, words))) = best {
        let share = score / total;
        let shown: Vec<&str> = words.iter().take(4).map(String::as_str).collect();
        return Some(
            Field::new(sig.to_owned())
                .with_confidence((0.2 + 0.5 * share).min(0.7))
                .with_evidence(format!("keywords: {}", shown.join(", "))),
        );
    }

    // 3. Nothing to go on: no answer.
    None
}

/// With `base_rates`, when nothing points at a SIG: the most common one.
fn area_base_rate(repo: &Repo) -> Option<Field<String>> {
    repo.labels
        .iter()
        .any(|l| l.name == AREA_FALLBACK)
        .then(|| {
            Field::new(AREA_FALLBACK.to_owned())
                .with_confidence(0.15)
                .with_evidence("no signal; base rate: the most common SIG")
        })
}

/// A version number, not just the template's `$ kubectl version` prompt.
static VERSION_STRING: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?i)\bv?1\.\d{1,2}\.\d+\b|\bv1\.\d{1,2}\b|gitversion:\s*"v"#).unwrap()
});
static NUMBERED_STEP: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^\s*(?:\d+[.)]|[-*] )\s*\S").unwrap());
static STACK_TRACE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)goroutine \d+ \[|panic:|\.go:\d+|^[EWIF]\d{4} \d\d:\d\d:\d\d|traceback \(most recent|exception in thread|level=error|\berror:").unwrap()
});
static REPRO_SNIPPET: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)apiversion:|^\s*\$?\s*kubectl |^\s*\$ |^\s*kind: ").unwrap());

const ENVIRONMENT_HEADINGS: &[&str] = &[
    "cloud provider",
    "os version",
    "install tools",
    "container runtime",
    "related plugins",
    "environment",
];
const ENVIRONMENT_WORDS: &[&str] = &[
    "ubuntu",
    "debian",
    "centos",
    "rhel",
    "red hat",
    "fedora",
    "amazon linux",
    "bottlerocket",
    "flatcar",
    "eks",
    "gke",
    "aks",
    "minikube",
    "kind cluster",
    "k3s",
    "kubeadm",
    "containerd",
    "cri-o",
    "docker",
    "uname",
    "kernel",
];

fn reproducibility(doc: &Doc) -> Field<Reproducibility> {
    let text = &doc.text;
    let has_version = doc
        .filled(&["kubernetes version"])
        .is_some_and(|s| VERSION_STRING.is_match(&s.to_lowercase()))
        || VERSION_STRING.is_match(text);
    let has_environment = doc.filled(ENVIRONMENT_HEADINGS).is_some()
        || contains_any(text, ENVIRONMENT_WORDS).is_some();
    let repro_section = doc.filled(&["how can we reproduce", "steps to reproduce", "reproduce"]);
    let has_steps = repro_section.is_some()
        || text.contains("steps to reproduce")
        || NUMBERED_STEP.find_iter(text).count() >= 3;
    let has_expected_actual = (doc.filled(&["what happened"]).is_some()
        && doc.filled(&["what did you expect", "expected"]).is_some())
        || (text.contains("expected")
            && contains_any(text, &["actual", "instead", " but "]).is_some());
    let has_minimal_repro = repro_section
        .is_some_and(|s| s.contains("```") || REPRO_SNIPPET.is_match(&s.to_lowercase()));
    let has_logs = STACK_TRACE.is_match(text) || (doc.code_blocks > 0 && text.contains("log"));
    // Lenient on purpose: in dev-split bugs 85% have enough information, and
    // v1's stricter rule (steps and expected/actual and version or
    // environment) was wrong on most reports it called insufficient.
    let enough_info = has_steps || has_expected_actual || has_minimal_repro || has_logs;
    Field::new(Reproducibility::from_checks(
        has_version,
        has_environment,
        has_steps,
        has_expected_actual,
        has_minimal_repro,
        has_logs,
        enough_info,
    ))
    .with_confidence(0.6)
    .with_evidence("template sections, code blocks and version strings")
}

/// With `base_rates`: the most common priority for each kind among dev-split
/// issues that have one, and its share there as the confidence. Nothing in an
/// issue's text predicted its priority better (panics, "data loss",
/// release-blocking jobs and the like were each spread over all four
/// levels), so without `base_rates` severity is left out. Questions, spam and
/// "other" get none.
fn severity_base_rate(kind: Kind) -> Option<Field<Severity>> {
    let (severity, share, of) = match kind {
        Kind::FailingTest => (Severity::ImportantSoon, 0.58, "failing tests"),
        Kind::Flake => (Severity::ImportantSoon, 0.54, "flaky tests"),
        Kind::Bug => (Severity::ImportantLongterm, 0.37, "bugs"),
        Kind::Feature => (Severity::ImportantLongterm, 0.44, "feature requests"),
        Kind::Cleanup => (Severity::ImportantLongterm, 0.52, "cleanups"),
        Kind::Docs => (Severity::Backlog, 0.6, "docs issues"),
        Kind::Question | Kind::Other | Kind::Spam => return None,
    };
    Some(
        Field::new(severity)
            .with_confidence(share)
            .with_evidence(format!("base rate: the most common priority for {of}")),
    )
}

/// A Kubernetes version in the title after "in", "since", "on" and the
/// like: "kubectl wait fails in 1.31.0-rc.1", "(v1.31)" doesn't count.
static TITLE_VERSION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b(?:in|since|starting in|starting with|after|on|with)\s+(?:k8s\s+|kubernetes\s+)?v?1\.\d{2}\b").unwrap()
});
/// An upgrade between two versions: "upgraded from v1.24.6 to v1.26.4",
/// "1.24 -> 1.25".
static UPGRADE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?:upgrad|updat|migrat)\w*\s+(?:\S+\s+){0,4}?from\s+(?:kubernetes\s+|k8s\s+)?v?1\.\d+(?:\.\d+)?\s+to\s+v?1\.\d+|\bv?1\.\d{2}(?:\.\d+)?\s*(?:->|→)\s*v?1\.\d{2}\b").unwrap()
});

fn is_regression(doc: &Doc) -> Field<bool> {
    const PHRASES: &[&str] = &[
        "this is a regression",
        "is a regression",
        "[regression]",
        "regression in v1.",
        "regression in 1.",
        "regressed in",
    ];
    let title = doc.title.to_lowercase();
    let evidence = if let Some(m) = TITLE_VERSION.find(&title) {
        Some(format!("title names the version: {:?}", m.as_str()))
    } else if let Some(m) = UPGRADE.find(&doc.text) {
        Some(format!("upgrade between versions: {:?}", m.as_str()))
    } else {
        contains_any(&doc.text, PHRASES).map(|p| format!("phrase {p:?}"))
    };
    match evidence {
        Some(evidence) => Field::new(true)
            .with_confidence(0.4)
            .with_evidence(evidence),
        None => Field::new(false)
            .with_confidence(0.8)
            .with_evidence("no version-specific breakage stated"),
    }
}

static CVE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\bcve-\d{4}-\d+").unwrap());

fn security_sensitive(doc: &Doc) -> Field<bool> {
    const PHRASES: &[&str] = &[
        "vulnerability",
        "vulnerabilities",
        "exploit",
        "privilege escalation",
        "escalate privileges",
        "security issue",
        "security bug",
        "security vulnerability",
        "remote code execution",
        "container escape",
        "container breakout",
        "bypass authorization",
        "bypass authentication",
        "leaks secret",
        "secret leak",
        "credentials leak",
    ];
    if let Some(m) = CVE.find(&doc.text) {
        return Field::new(true)
            .with_confidence(0.7)
            .with_evidence(format!("mentions {}", m.as_str().to_uppercase()));
    }
    match contains_any(&doc.text, PHRASES) {
        Some(p) => Field::new(true)
            .with_confidence(0.6)
            .with_evidence(format!("phrase {p:?}")),
        None => Field::new(false)
            .with_confidence(0.8)
            .with_evidence("no security keyword"),
    }
}

/// A Kubernetes version: `major.minor[.patch]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct K8sVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: Option<u32>,
}

impl K8sVersion {
    /// Parses `v1.33.2`, `1.33` and the like, ignoring a pre-release suffix.
    pub fn parse(s: &str) -> Option<K8sVersion> {
        let s = s.trim().trim_start_matches('v');
        let core = s.split(['-', '+']).next()?;
        let mut parts = core.split('.');
        let major = parts.next()?.parse().ok()?;
        let minor = parts.next()?.parse().ok()?;
        let patch = parts.next().and_then(|p| p.parse().ok());
        Some(K8sVersion {
            major,
            minor,
            patch,
        })
    }

    /// Whether `self` is older than `latest`. Without a patch on both sides
    /// only the minor versions are compared.
    pub fn older_than(&self, latest: &K8sVersion) -> bool {
        match (self.patch, latest.patch) {
            (Some(a), Some(b)) => (self.major, self.minor, a) < (latest.major, latest.minor, b),
            _ => (self.major, self.minor) < (latest.major, latest.minor),
        }
    }
}

static SERVER_GIT_VERSION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?is)server version:.{0,80}?gitversion:\s*"v(\d+\.\d+\.\d+)"#).unwrap()
});
static SERVER_VERSION: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)server version:\s*v?(\d+\.\d+(?:\.\d+)?)").unwrap());
static GIT_VERSION: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"(?i)gitversion:\s*"v(\d+\.\d+\.\d+)"#).unwrap());
static BARE_VERSION: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\bv?(1\.\d{1,2}(?:\.\d{1,3})?)\b").unwrap());
static NAMED_VERSION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(?:kubernetes|k8s|kubelet|kubectl|kube-apiserver)\s+v?(1\.\d{1,2}(?:\.\d{1,3})?)\b",
    )
    .unwrap()
});
static V_VERSION: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\bv(1\.\d{1,2}(?:\.\d{1,3})?)\b").unwrap());

/// The Kubernetes version the issue reports, and whether it came from a
/// reliable place (the server version, or the template's version section).
pub fn reported_version(title: &str, body: &str) -> Option<(K8sVersion, bool)> {
    let doc = Doc::parse(title, body);
    find_version(&doc)
}

fn find_version(doc: &Doc) -> Option<(K8sVersion, bool)> {
    let first = |re: &Regex, text: &str| {
        re.captures(text)
            .and_then(|c| K8sVersion::parse(c.get(1)?.as_str()))
    };
    let body = doc
        .sections
        .iter()
        .map(|(_, c)| c.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    if let Some(v) = first(&SERVER_GIT_VERSION, &body).or_else(|| first(&SERVER_VERSION, &body)) {
        return Some((v, true));
    }
    if let Some(section) = doc.filled(&["kubernetes version"])
        && let Some(v) = first(&GIT_VERSION, section).or_else(|| first(&BARE_VERSION, section))
    {
        return Some((v, true));
    }
    first(&GIT_VERSION, &body)
        .or_else(|| first(&NAMED_VERSION, &body))
        .or_else(|| first(&V_VERSION, &body))
        .map(|v| (v, false))
}

fn outdated_version(doc: &Doc, repo: &Repo) -> Option<Field<bool>> {
    let latest = K8sVersion::parse(repo.latest_release.as_deref()?)?;
    let (reported, reliable) = find_version(doc)?;
    let outdated = reported.older_than(&latest);
    let shown = match reported.patch {
        Some(p) => format!("v{}.{}.{p}", reported.major, reported.minor),
        None => format!("v{}.{}", reported.major, reported.minor),
    };
    Some(
        Field::new(outdated)
            .with_confidence(if reliable { 0.7 } else { 0.45 })
            .with_evidence(format!(
                "reported {shown}, latest {}",
                repo.latest_release.as_deref().unwrap_or_default()
            )),
    )
}

static MULTIPLE_PROBLEMS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b(?:another|second|separate|unrelated|additional) (?:issue|problem|bug)\b|\b(?:issue|problem|bug) (?:#?2|two)\b|\b2nd (?:issue|problem)\b|\balso noticed\b").unwrap()
});

fn scope(doc: &Doc) -> Field<Scope> {
    match MULTIPLE_PROBLEMS.find(&doc.text) {
        Some(m) => Field::new(Scope::Multiple)
            .with_confidence(0.5)
            .with_evidence(format!("phrase {:?}", m.as_str())),
        None => Field::new(Scope::Single)
            .with_confidence(0.6)
            .with_evidence("no sign of a second problem"),
    }
}

/// One point per checklist item, mapped onto the 1-5 rubric.
fn quality(doc: &Doc, kind: Option<Kind>) -> Field<u8> {
    let filled_sections = doc
        .sections
        .iter()
        .filter(|(h, c)| !h.is_empty() && meaningful(c))
        .count();
    let all_sections = doc.sections.iter().filter(|(h, _)| !h.is_empty()).count();
    let mut checks: Vec<&str> = Vec::new();
    if doc.title.trim().len() >= 20 {
        checks.push("descriptive title");
    }
    if doc.body_len >= 300 {
        checks.push("detailed body");
    }
    if all_sections > 0 && filled_sections * 3 >= all_sections * 2 {
        checks.push("template filled in");
    }
    if doc.code_blocks > 0 || STACK_TRACE.is_match(&doc.text) {
        checks.push("code or logs");
    }
    let actionable = match kind {
        Some(Kind::Bug) => {
            doc.filled(&["how can we reproduce"]).is_some()
                && doc.filled(&["what did you expect"]).is_some()
        }
        Some(Kind::Feature) => doc.filled(&["why is this needed"]).is_some() || doc.body_len >= 600,
        Some(Kind::FailingTest | Kind::Flake) => doc
            .filled(&["testgrid link", "reason for failure"])
            .is_some(),
        _ => doc.body_len >= 600,
    };
    if actionable {
        checks.push("actionable");
    }
    let score = if doc.body_len < 30 {
        1
    } else {
        checks.len().clamp(1, 5) as u8
    };
    let evidence = if checks.is_empty() {
        "none of the checklist items".to_owned()
    } else {
        checks.join(", ")
    };
    Field::new(score)
        .with_confidence(0.4)
        .with_evidence(evidence)
}

#[cfg(test)]
mod tests {
    use super::*;
    use triage_core::{Label, ReproItem};

    fn repo() -> Repo {
        let labels = [
            "sig/node",
            "sig/network",
            "sig/cli",
            "sig/storage",
            "kind/bug",
        ]
        .into_iter()
        .map(|name| Label {
            name: name.into(),
            description: None,
        })
        .collect();
        Repo {
            name: "kubernetes/kubernetes".into(),
            labels,
            latest_release: Some("v1.33.2".into()),
            as_of: "2025-07-01T00:00:00Z".into(),
        }
    }

    fn issue(title: &str, body: &str) -> Issue {
        Issue {
            number: 1,
            title: title.into(),
            body: body.into(),
            author_association: "NONE".into(),
            created_at: "2025-07-01T00:00:00Z".into(),
        }
    }

    const BUG: &str = r#"### What happened?

After upgrading from v1.32.4 to v1.33.1 the kubelet keeps restarting pods with OOMKilled even though memory usage is low.

### What did you expect to happen?

Pods keep running as they did on v1.32.

### How can we reproduce it (as minimally and precisely as possible)?

1. Create a pod:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: test
```

2. Wait five minutes.

### Anything else we need to know?

```
E0612 10:11:12.123456    1234 kubelet.go:2345] "Error syncing pod" err="OOMKilled"
```

### Kubernetes version

<details>

```console
$ kubectl version
Client Version: v1.33.1
Server Version: v1.33.1
```

</details>

### Cloud provider

<details>
none, bare metal
</details>

### OS version

<details>

```console
# On Linux:
$ cat /etc/os-release
Ubuntu 24.04
```

</details>
"#;

    #[test]
    fn bug_template() {
        let t = triage(
            &issue("kubelet restarts pods with OOMKilled after upgrade", BUG),
            &repo(),
            &HeuristicConfig::default(),
        );
        let kind = t.kind.unwrap();
        assert_eq!(kind.value, Kind::Bug);
        assert_eq!(kind.evidence.as_deref(), Some("bug report template"));
        assert_eq!(t.area.unwrap().value, "sig/node");
        assert!(t.is_regression.unwrap().value);
        assert!(!t.security_sensitive.unwrap().value);
        let repro = t.reproducibility.unwrap().value;
        assert!(repro.has_version && repro.has_steps && repro.has_expected_actual);
        assert!(repro.has_minimal_repro && repro.has_logs && repro.has_environment);
        assert!(repro.enough_info);
        assert!(repro.missing.is_empty(), "{:?}", repro.missing);
        // v1.33.1 < v1.33.2.
        let outdated = t.outdated_version.unwrap();
        assert!(outdated.value);
        assert_eq!(outdated.confidence, Some(0.7));
        assert!(t.quality.unwrap().value >= 4);
        // Nothing in the text says how urgent it is.
        assert!(t.severity.is_none());
        assert!(t.duplicates.is_none() && t.suggested_reply.is_none());
    }

    #[test]
    fn empty_bug_template_lacks_info() {
        let body = "### What happened?\n\nit broke\n\n### What did you expect to happen?\n\n_No response_\n\n### How can we reproduce it (as minimally and precisely as possible)?\n\n_No response_\n\n### Kubernetes version\n\n<details>\n\n```console\n$ kubectl version\n# paste output here\n```\n\n</details>\n";
        let t = triage(&issue("broken", body), &repo(), &HeuristicConfig::default());
        let repro = t.reproducibility.unwrap().value;
        assert!(!repro.has_steps && !repro.has_version && !repro.enough_info);
        assert!(repro.missing.contains(&ReproItem::Steps));
        assert!(t.outdated_version.is_none());
        assert!(t.quality.unwrap().value <= 2);
    }

    #[test]
    fn templates_decide_kind() {
        let cases = [
            (
                "### What would you like to be added?\n\nA flag.\n\n### Why is this needed?\n\nBecause.",
                Kind::Feature,
            ),
            (
                "### Which jobs are failing?\n\nci-kubernetes-e2e\n\n### Which tests are failing?\n\nfoo",
                Kind::FailingTest,
            ),
            (
                "### Which jobs are flaking?\n\nci-kubernetes-e2e\n\n### Which tests are flaking?\n\nfoo",
                Kind::Flake,
            ),
        ];
        for (body, expected) in cases {
            let k = kind(&Doc::parse("something", body)).unwrap();
            assert_eq!(k.value, expected, "{body}");
            assert!(k.confidence.unwrap() >= 0.75);
        }
        let k = kind(&Doc::parse(
            "[Flaky Test] TestFoo times out",
            "see testgrid",
        ))
        .unwrap();
        assert_eq!(k.value, Kind::Flake);
        // v1 missed "flaking".
        let k = |title: &str, body: &str| kind(&Doc::parse(title, body)).unwrap().value;
        assert_eq!(k("[Flaking Test] [sig-node] foo", "x"), Kind::Flake);
        assert_eq!(
            k("Failure cluster [3aa867c5...] flakiness in job", "x"),
            Kind::FailingTest
        );
        // The failing-test template's comment about /kind flake is not a
        // flake marker (HTML comments are dropped).
        assert_eq!(
            k(
                "TestFoo fails",
                "### Which jobs are failing?\n\n<!-- If this is a flake, please add: /kind flake -->\nci-kubernetes-e2e-gce\n"
            ),
            Kind::FailingTest
        );
    }

    #[test]
    fn keywords_decide_kind_without_template() {
        let k = |title: &str, body: &str| kind(&Doc::parse(title, body)).unwrap().value;
        assert_eq!(
            k("How to set a pod's hostname?", "I tried things."),
            Kind::Question
        );
        assert_eq!(
            k("Add support for IPv6 in foo", "It would help."),
            Kind::Feature
        );
        assert_eq!(
            k("Cleanup: remove unused helper", "dead code"),
            Kind::Cleanup
        );
        assert_eq!(
            k("[FG:InPlacePodVerticalScaling] Handle NRI plugins", "x"),
            Kind::Feature
        );
        assert_eq!(
            k("Add unit test coverage for the converter", "x"),
            Kind::Cleanup
        );
        assert_eq!(k("Add a /readyz endpoint", "x"), Kind::Feature);
        assert_eq!(k("kube-proxy panics", "panic: runtime error"), Kind::Bug);
        // No rule fires: no kind, rather than the base rate.
        assert!(kind(&Doc::parse("hello", "world")).is_none());
    }

    #[test]
    fn area_only_from_repo_labels() {
        let doc = Doc::parse(
            "kubectl apply prints wrong diff",
            "kubectl diff output is off",
        );
        assert_eq!(area(&doc, &repo()).unwrap().value, "sig/cli");
        let doc = Doc::parse("Windows nodes fail", "hostprocess containers on windows");
        // sig/windows isn't one of the repo's labels here; `container` is sig/node.
        assert_eq!(area(&doc, &repo()).unwrap().value, "sig/node");
        // Nothing to go on: no area.
        assert!(area(&Doc::parse("hello", "world"), &repo()).is_none());
    }

    #[test]
    fn test_tags_decide_area() {
        let doc = Doc::parse(
            "[Flaking Test] [sig-storage] CSI Volumes should mount",
            "kubelet logs show the pod",
        );
        let a = area(&doc, &repo()).unwrap();
        assert_eq!(a.value, "sig/storage");
        assert_eq!(a.evidence.as_deref(), Some("test tag [sig-storage]"));
        // Tags are matched on letters only, and only to the repo's labels.
        assert_eq!(sig_label(&repo(), "net-work"), Some("sig/network"));
        assert_eq!(sig_label(&repo(), "apimachinery"), None);
    }

    #[test]
    fn mined_tokens_count_for_area() {
        // "ipvs" and "conntrack" are only in the mined table, not in SIG_KEYWORDS.
        let doc = Doc::parse("ipvs rules go stale", "conntrack entries stay");
        let a = area(&doc, &repo()).unwrap();
        assert_eq!(a.value, "sig/network");
        assert!(a.evidence.unwrap().contains("ipvs"));
        // Tokens split like the mining script: no @mentions, no number-ish words.
        let t = tokens("@liggitt see kep-5278 and e2e_node in v1.33");
        assert!(t.contains("e2e_node") && t.contains("see"));
        assert!(!t.contains("liggitt") && !t.contains("kep-5278"));
    }

    #[test]
    fn no_signal_leaves_fields_out() {
        let t = triage(
            &issue("hello", "world"),
            &repo(),
            &HeuristicConfig::default(),
        );
        assert!(t.kind.is_none() && t.area.is_none() && t.severity.is_none());
        // An unknown kind still gets the checklist, as in the other strategies.
        assert!(t.reproducibility.is_some());
        assert!(t.quality.is_some() && t.scope.is_some());
    }

    #[test]
    fn base_rates_fill_what_the_text_leaves_out() {
        let config = HeuristicConfig { base_rates: true };
        let t = triage(&issue("hello", "world"), &repo(), &config);
        let kind = t.kind.unwrap();
        assert_eq!((kind.value, kind.confidence), (Kind::Bug, Some(0.3)));
        let area = t.area.unwrap();
        assert_eq!(area.value, "sig/node");
        assert!(area.confidence.unwrap() < 0.2);
        assert!(area.evidence.unwrap().contains("base rate"));
        assert_eq!(t.severity.unwrap().value, Severity::ImportantLongterm);
        // A text signal still wins over the base rate.
        let t = triage(
            &issue(
                "kubectl apply prints wrong diff",
                "kubectl diff output is off",
            ),
            &repo(),
            &config,
        );
        assert_eq!(t.area.unwrap().value, "sig/cli");

        assert_eq!(
            severity_base_rate(Kind::FailingTest).unwrap().value,
            Severity::ImportantSoon
        );
        assert_eq!(
            severity_base_rate(Kind::Docs).unwrap().value,
            Severity::Backlog
        );
        assert!(severity_base_rate(Kind::Question).is_none());
        let mut no_node = repo();
        no_node.labels.retain(|l| l.name != "sig/node");
        assert!(area_base_rate(&no_node).is_none());
    }

    #[test]
    fn security_keywords() {
        let t = |body: &str| security_sensitive(&Doc::parse("x", body)).value;
        assert!(t("This is CVE-2025-1234 in the kubelet"));
        assert!(t("allows privilege escalation to root"));
        assert!(!t("set securityContext.runAsUser"));
    }

    #[test]
    fn versions() {
        let v = |body: &str| {
            reported_version("t", body).map(|(v, reliable)| (v.minor, v.patch, reliable))
        };
        assert_eq!(
            v("Server Version: v1.30.4\nClient Version: v1.31.0"),
            Some((30, Some(4), true))
        );
        assert_eq!(
            v(r#"Server Version: version.Info{Major:"1", Minor:"27", GitVersion:"v1.27.3"}"#),
            Some((27, Some(3), true))
        );
        assert_eq!(
            v("### Kubernetes version\n\n1.29\n"),
            Some((29, None, true))
        );
        assert_eq!(
            v("we run kubernetes 1.28.2 with containerd 1.7.2"),
            Some((28, Some(2), false))
        );
        assert_eq!(v("apiVersion: apps/v1 and go1.22.1"), None);

        let latest = K8sVersion::parse("v1.33.2").unwrap();
        assert!(K8sVersion::parse("1.32.9").unwrap().older_than(&latest));
        assert!(!K8sVersion::parse("1.33").unwrap().older_than(&latest));
        assert!(
            !K8sVersion::parse("v1.34.0-alpha.1")
                .unwrap()
                .older_than(&latest)
        );
    }

    #[test]
    fn scope_and_regression_phrases() {
        assert_eq!(
            scope(&Doc::parse("x", "Also, another issue: the logs")).value,
            Scope::Multiple
        );
        assert_eq!(scope(&Doc::parse("x", "one thing")).value, Scope::Single);
        let r = |title: &str, body: &str| is_regression(&Doc::parse(title, body)).value;
        assert!(r("kubectl wait fails in 1.31.0-rc.1", "x"));
        assert!(r("x", "we upgraded the cluster from v1.24.6 to v1.26.4"));
        assert!(r("x", "1.24 -> 1.25 broke it"));
        assert!(r("x", "This is a regression."));
        // Too loose: wrong 9 times out of 10 on dev-split issues.
        assert!(!r("x", "after upgrading the pods restart"));
        assert!(!r("x", "this worked in 1.31"));
        assert!(!r("x", "new cluster"));
    }
}
