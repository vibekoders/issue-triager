use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

/// A short, stable hash of a strategy config, as it was sent.
///
/// It hashes canonical JSON: object keys sorted, no whitespace, non-ASCII
/// characters unescaped. Python produces the same bytes with
/// `json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`,
/// so the harness can compute the hash before calling the service (it is part
/// of the Restate workflow key).
pub fn config_hash(config: &serde_json::Value) -> String {
    let canonical = canonical(config).to_string();
    let digest = Sha256::digest(canonical.as_bytes());
    digest[..6].iter().map(|b| format!("{b:02x}")).collect()
}

/// `value` with every object's keys in sorted order. The workspace enables
/// serde_json's `preserve_order` (for schema property order), so a `Map`
/// keeps insertion order and `to_string` would not sort on its own. Rust
/// compares strings by their UTF-8 bytes, which is the same order as
/// Python's code-point order.
fn canonical(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            let sorted: Map<String, Value> = entries
                .into_iter()
                .map(|(k, v)| (k.clone(), canonical(v)))
                .collect();
            Value::Object(sorted)
        }
        Value::Array(items) => Value::Array(items.iter().map(canonical).collect()),
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_order_does_not_matter() {
        let a: serde_json::Value =
            serde_json::from_str(r#"{"b":1,"a":{"y":0.8,"x":"é"}}"#).unwrap();
        let b: serde_json::Value =
            serde_json::from_str(r#"{"a":{"x":"é","y":0.8},"b":1}"#).unwrap();
        assert_eq!(config_hash(&a), config_hash(&b));
    }

    /// Built by insertion, in an order that is not sorted, including objects
    /// nested in an array: the hash must not depend on whether serde_json
    /// keeps insertion order.
    #[test]
    fn insertion_order_does_not_matter() {
        let mut inner = Map::new();
        inner.insert("model".into(), "ollama:qwen3:1.7b".into());
        inner.insert("kind".into(), "llm".into());
        let mut a = Map::new();
        a.insert("voters".into(), Value::Array(vec![Value::Object(inner)]));
        a.insert("aggregator".into(), serde_json::json!({"kind": "vote"}));
        let b = serde_json::json!({
            "aggregator": {"kind": "vote"},
            "voters": [{"kind": "llm", "model": "ollama:qwen3:1.7b"}],
        });
        assert_eq!(config_hash(&Value::Object(a.clone())), config_hash(&b));
        assert_eq!(
            canonical(&Value::Object(a)).to_string(),
            r#"{"aggregator":{"kind":"vote"},"voters":[{"kind":"llm","model":"ollama:qwen3:1.7b"}]}"#
        );
    }

    /// Pinned against Python:
    /// `hashlib.sha256(json.dumps({"model": "ollama:qwen3:4b"}, sort_keys=True,
    /// separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()[:12]`
    #[test]
    fn matches_python() {
        let v = serde_json::json!({"model": "ollama:qwen3:4b"});
        assert_eq!(config_hash(&v), PYTHON_HASH);
        // `{}`, also pinned in the harness tests.
        assert_eq!(config_hash(&serde_json::json!({})), "44136fa355b3");
    }

    const PYTHON_HASH: &str = "aa413933d122";
}
