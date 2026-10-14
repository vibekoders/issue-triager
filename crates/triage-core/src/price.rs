/// Prices in USD per million tokens.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Price {
    pub input_per_mtok: f64,
    pub output_per_mtok: f64,
}

impl Price {
    pub const fn new(input_per_mtok: f64, output_per_mtok: f64) -> Self {
        Price {
            input_per_mtok,
            output_per_mtok,
        }
    }

    pub fn cost(&self, input_tokens: u64, output_tokens: u64) -> f64 {
        (input_tokens as f64 * self.input_per_mtok + output_tokens as f64 * self.output_per_mtok)
            / 1_000_000.0
    }
}

/// Prices by model address (`provider:model`), matched by prefix, first match
/// wins. Check them against the providers' price pages before reporting.
///
/// Local Ollama models cost nothing to call, which would make the cost axis
/// meaningless. They get shadow prices instead: what a hosted model of about
/// the same size costs.
const PRICES: &[(&str, Price)] = &[
    // Ollama decision models, shadow-priced like Jev (output is free).
    ("ollama:nimble", Price::new(0.042, 0.0)),
    ("ollama:tev1", Price::new(0.042, 0.0)),
    // Ollama chat models, shadow-priced by size.
    ("ollama:qwen3:0.6b", Price::new(0.02, 0.08)),
    ("ollama:qwen3:1.7b", Price::new(0.04, 0.15)),
    ("ollama:qwen3:4b", Price::new(0.06, 0.25)),
    ("ollama:qwen3:8b", Price::new(0.10, 0.40)),
    ("ollama:qwen3:14b", Price::new(0.15, 0.60)),
    ("ollama:llama3.2:1b", Price::new(0.03, 0.10)),
    ("ollama:llama3.2:3b", Price::new(0.05, 0.20)),
    ("ollama:", Price::new(0.10, 0.40)),
    // Hosted.
    ("typesafe:", Price::new(0.042, 0.0)),
    // Azure deployments are addressed by deployment name. Global Standard,
    // short context (requests here are far below the long-context cutoff).
    // Cached input is cheaper still; it isn't counted separately.
    ("azure:gpt-6-luna", Price::new(0.10, 0.50)),
    ("azure:gpt-6-sol", Price::new(2.0, 10.0)),
    // OpenAI's API, standard tier, short context; the same models as the
    // Azure deployments above, at the same prices.
    ("openai:gpt-6-luna", Price::new(0.10, 0.50)),
    ("openai:gpt-6-sol", Price::new(2.0, 10.0)),
    ("anthropic:claude-haiku-4-5", Price::new(1.0, 5.0)),
    ("anthropic:claude-sonnet-5", Price::new(2.0, 10.0)),
    ("anthropic:claude-opus-5", Price::new(5.0, 25.0)),
    // Gemini API, paid tier, prompts up to 200k tokens.
    ("gemini:gemini-3.5-flash-lite", Price::new(0.30, 2.50)),
    ("gemini:gemini-3.8-flash", Price::new(0.75, 3.75)),
    ("gemini:gemini-3.1-pro-preview", Price::new(2.0, 12.0)),
];

/// The price of a model address, if it's in the table.
pub fn price(model: &str) -> Option<Price> {
    PRICES
        .iter()
        .find(|(prefix, _)| model.starts_with(prefix))
        .map(|(_, price)| *price)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specific_prefix_wins() {
        assert_eq!(price("ollama:qwen3:4b"), Some(Price::new(0.06, 0.25)));
        assert_eq!(price("ollama:mistral"), Some(Price::new(0.10, 0.40)));
        assert_eq!(price("unknown:model"), None);
    }

    /// Every chat and decision model of the hosted experiment profiles
    /// (evals/experiments/profiles.yaml) has a price.
    #[test]
    fn hosted_profile_models_are_priced() {
        for (model, input, output) in [
            ("openai:gpt-6-luna", 0.10, 0.50),
            ("openai:gpt-6-sol", 2.0, 10.0),
            ("azure:gpt-6-luna", 0.10, 0.50),
            ("azure:gpt-6-sol", 2.0, 10.0),
            ("anthropic:claude-haiku-4-5", 1.0, 5.0),
            ("anthropic:claude-sonnet-5", 2.0, 10.0),
            ("gemini:gemini-3.5-flash-lite", 0.30, 2.50),
            ("gemini:gemini-3.8-flash", 0.75, 3.75),
            ("gemini:gemini-3.1-pro-preview", 2.0, 12.0),
            ("typesafe:jev-latest", 0.042, 0.0),
        ] {
            assert_eq!(price(model), Some(Price::new(input, output)), "{model}");
        }
    }

    #[test]
    fn cost() {
        assert!((Price::new(1.0, 5.0).cost(1_000_000, 100_000) - 1.5).abs() < 1e-12);
    }
}
