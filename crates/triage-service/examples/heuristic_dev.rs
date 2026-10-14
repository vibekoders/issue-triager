//! Scores the heuristic offline on normalized records, for tuning it without
//! Restate or the harness:
//!
//!   cargo run -q -p triage-service --example heuristic_dev -- FILE.jsonl [--errors FIELD] [--base-rates]
//!
//! FILE.jsonl holds records as `data/normalized/<repo>.jsonl` writes them
//! (`{issue, repo, expected, ...}`). Feed it dev-split records only: the
//! heuristic is tuned on dev, and the test split is for experiment reports.
//! `scripts/mine_sig_keywords.py --write-splits DIR` writes the dev-split
//! files used for heuristic v2 (dev minus dev-200 for tuning, dev-200 held
//! out).
//!
//! The scores follow the harness's evaluators: a field is scored when the
//! record has an expected answer, accuracy is over the attempted cases, and
//! the composite counts an unattempted field as 0 (judge-free weights from
//! `evals/suite.yaml`). `--errors FIELD` prints the misses for one field;
//! `--base-rates` scores the heuristic with `base_rates: true`.

use std::collections::BTreeMap;
use std::io::BufRead;

use serde::Deserialize;
use triage_core::config::HeuristicConfig;
use triage_core::{Issue, Repo};
use triage_service::strategies::heuristic;

#[derive(Deserialize)]
struct Record {
    issue: Issue,
    repo: Repo,
    expected: Expected,
}

#[derive(Deserialize)]
struct Expected {
    kind: Option<Vec<String>>,
    area: Option<Vec<String>>,
    severity: Option<String>,
    is_regression: Option<bool>,
    security_sensitive: Option<bool>,
    enough_info: Option<bool>,
}

#[derive(Default)]
struct Score {
    expected: usize,
    attempted: usize,
    correct: usize,
    /// For booleans: (predicted true, actually true, both).
    pos: (usize, usize, usize),
}

impl Score {
    fn add(&mut self, attempted: bool, correct: bool) {
        self.expected += 1;
        if attempted {
            self.attempted += 1;
            if correct {
                self.correct += 1;
            }
        }
    }

    fn add_bool(&mut self, predicted: Option<bool>, actual: bool) {
        self.add(predicted.is_some(), predicted == Some(actual));
        if let Some(p) = predicted {
            self.pos.0 += usize::from(p);
            self.pos.1 += usize::from(actual);
            self.pos.2 += usize::from(p && actual);
        }
    }

    /// Mean score over the expected cases, unattempted counting 0.
    fn mean(&self) -> f64 {
        if self.expected == 0 {
            0.0
        } else {
            self.correct as f64 / self.expected as f64
        }
    }

    fn line(&self, name: &str) -> String {
        let acc = if self.attempted == 0 {
            "not attempted".to_owned()
        } else {
            format!(
                "{:5.1}%",
                100.0 * self.correct as f64 / self.attempted as f64
            )
        };
        let mut s = format!(
            "{name:<19} {acc:>13}  {}/{} attempted",
            self.attempted, self.expected
        );
        let (p, a, both) = self.pos;
        if p + a > 0 {
            let pct = |n: usize, d: usize| {
                if d == 0 {
                    "-".to_owned()
                } else {
                    format!("{:.0}%", 100.0 * n as f64 / d as f64)
                }
            };
            s += &format!(
                "  P {} / R {} (predicted {p}, actual {a})",
                pct(both, p),
                pct(both, a)
            );
        }
        s
    }
}

const SEVERITY_ORDER: [&str; 4] = [
    "critical",
    "important_soon",
    "important_longterm",
    "backlog",
];

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args.first().ok_or_else(|| {
        anyhow::anyhow!("usage: heuristic_dev FILE.jsonl [--errors FIELD] [--base-rates]")
    })?;
    let errors_for = args
        .iter()
        .position(|a| a == "--errors")
        .and_then(|i| args.get(i + 1))
        .cloned();
    let config = HeuristicConfig {
        base_rates: args.iter().any(|a| a == "--base-rates"),
    };

    let mut scores: BTreeMap<&str, Score> = BTreeMap::new();
    let mut kind_confusion: BTreeMap<(String, String), usize> = BTreeMap::new();
    let mut severity_distance = (0usize, 0usize);
    let mut cases = 0;
    let file = std::io::BufReader::new(std::fs::File::open(path)?);
    for line in file.lines() {
        let record: Record = serde_json::from_str(&line?)?;
        let t = heuristic::triage(&record.issue, &record.repo, &config);
        let e = &record.expected;
        cases += 1;
        let miss = |field: &str, predicted: String, expected: String| {
            if errors_for.as_deref() == Some(field) {
                println!(
                    "#{} [{field}] predicted {predicted}, expected {expected}: {}",
                    record.issue.number, record.issue.title
                );
            }
        };

        if let Some(kinds) = &e.kind {
            let p = t.kind.as_ref().map(|k| k.value.as_str().to_owned());
            let ok = p.as_ref().is_some_and(|p| kinds.contains(p));
            scores.entry("kind").or_default().add(p.is_some(), ok);
            let shown = p.clone().unwrap_or_else(|| "-".into());
            *kind_confusion
                .entry((kinds[0].clone(), shown.clone()))
                .or_default() += 1;
            if !ok {
                miss("kind", shown, kinds.join("|"));
            }
        }
        if let Some(areas) = &e.area {
            let p = t.area.as_ref().map(|a| a.value.clone());
            let ok = p.as_ref().is_some_and(|p| areas.contains(p));
            scores.entry("area").or_default().add(p.is_some(), ok);
            if !ok {
                miss("area", p.unwrap_or_else(|| "-".into()), areas.join("|"));
            }
        }
        if let Some(sev) = &e.severity {
            let p = t.severity.as_ref().map(|s| s.value.as_str().to_owned());
            let ok = p.as_deref() == Some(sev.as_str());
            scores.entry("severity").or_default().add(p.is_some(), ok);
            if let Some(p) = &p {
                let pos = |s: &str| SEVERITY_ORDER.iter().position(|x| *x == s).unwrap_or(0);
                severity_distance.0 += pos(p).abs_diff(pos(sev));
                severity_distance.1 += 1;
            }
            if !ok {
                miss("severity", p.unwrap_or_else(|| "-".into()), sev.clone());
            }
        }
        let bools = [
            (
                "is_regression",
                t.is_regression.as_ref().map(|f| f.value),
                e.is_regression,
            ),
            (
                "security_sensitive",
                t.security_sensitive.as_ref().map(|f| f.value),
                e.security_sensitive,
            ),
            (
                "enough_info",
                t.reproducibility.as_ref().map(|f| f.value.enough_info),
                e.enough_info,
            ),
        ];
        for (name, p, actual) in bools {
            if let Some(actual) = actual {
                scores.entry(name).or_default().add_bool(p, actual);
                if p != Some(actual) {
                    miss(
                        name,
                        p.map_or("-".into(), |b| b.to_string()),
                        actual.to_string(),
                    );
                }
            }
        }
    }

    println!("{cases} cases from {path}");
    let weights = [
        ("kind", 3.0),
        ("area", 2.0),
        ("enough_info", 2.0),
        ("severity", 1.0),
        ("is_regression", 1.0),
        ("security_sensitive", 1.0),
    ];
    for (name, _) in weights {
        if let Some(s) = scores.get(name) {
            println!("{}", s.line(name));
        }
    }
    if severity_distance.1 > 0 {
        println!(
            "severity mean distance {:.2}",
            severity_distance.0 as f64 / severity_distance.1 as f64
        );
    }
    let (sum, total) = weights
        .iter()
        .filter_map(|(n, w)| scores.get(n).map(|s| (w * s.mean(), w)))
        .fold((0.0, 0.0), |(a, b), (x, w)| (a + x, b + w));
    println!(
        "composite (no judge, no duplicates) {:.1}",
        100.0 * sum / total
    );
    println!("kind confusion (expected -> predicted: n):");
    for ((e, p), n) in &kind_confusion {
        if e != p {
            println!("  {e} -> {p}: {n}");
        }
    }
    Ok(())
}
