use std::fmt;

use crate::config::ModelsConfig;

/// Tier-based model selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    Fast,
    Primary,
    Smart,
}

impl fmt::Display for Tier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Tier::Fast => write!(f, "fast"),
            Tier::Primary => write!(f, "primary"),
            Tier::Smart => write!(f, "smart"),
        }
    }
}

pub struct ClassificationResult {
    pub tier: Tier,
    pub reason: String,
}

pub struct Router {
    models: ModelsConfig,
}

impl Router {
    pub fn new(models: ModelsConfig) -> Self {
        Self { models }
    }

    pub fn select(&self, tier: Tier) -> &str {
        match tier {
            Tier::Fast => &self.models.fast,
            Tier::Primary => &self.models.primary,
            Tier::Smart => &self.models.smart,
        }
    }

    /// Returns true when all three tier models resolve to the same string,
    /// meaning auto-routing would be pointless.
    pub fn is_uniform(&self) -> bool {
        self.models.fast == self.models.primary && self.models.primary == self.models.smart
    }
}

/// Classify a user query into a model tier using keyword/pattern heuristics.
pub fn classify_query(text: &str) -> ClassificationResult {
    let trimmed = text.trim();
    let lower = trimmed.to_lowercase();

    // --- Smart tier checks ---

    // Code fences
    if trimmed.contains("```") {
        return ClassificationResult {
            tier: Tier::Smart,
            reason: "contains code fence".to_string(),
        };
    }

    // Long messages (>500 chars)
    if trimmed.len() > 500 {
        return ClassificationResult {
            tier: Tier::Smart,
            reason: format!("long message ({} chars)", trimmed.len()),
        };
    }

    // 3+ question marks
    if trimmed.chars().filter(|&c| c == '?').count() >= 3 {
        return ClassificationResult {
            tier: Tier::Smart,
            reason: "multiple questions (3+ ?)".to_string(),
        };
    }

    // Smart keywords
    let smart_keywords = [
        "implement",
        "refactor",
        "debug",
        "analyze",
        "step by step",
        "write code",
        "architecture",
        "optimize",
        "algorithm",
        "explain how",
        "write a",
        "build a",
        "create a function",
        "design",
        "compare and contrast",
        "walk me through",
        "troubleshoot",
        "review this",
        "fix this",
        "rewrite",
    ];
    for kw in &smart_keywords {
        if lower.contains(kw) {
            return ClassificationResult {
                tier: Tier::Smart,
                reason: format!("keyword: {}", kw),
            };
        }
    }

    // --- Fast tier checks ---

    // Exact match greetings/acks (case-insensitive, trimmed)
    let fast_exact = [
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "ok",
        "okay",
        "yes",
        "no",
        "sure",
        "bye",
        "goodbye",
        "good morning",
        "good night",
        "gm",
        "gn",
        "yo",
        "sup",
        "ty",
        "thx",
        "np",
        "k",
        "yep",
        "nope",
        "nah",
        "yeah",
        "yea",
        "cool",
        "nice",
        "great",
        "awesome",
        "lol",
        "haha",
        "wow",
    ];
    if fast_exact.contains(&lower.as_str()) {
        return ClassificationResult {
            tier: Tier::Fast,
            reason: format!("greeting/ack: {}", lower),
        };
    }

    // Single-word messages
    let word_count = trimmed.split_whitespace().count();
    if word_count == 1 {
        return ClassificationResult {
            tier: Tier::Fast,
            reason: "single word".to_string(),
        };
    }

    // Note: short messages (2-3 words) fall through to Primary by default.
    // Short ≠ simple — "capital Ecuador" or "explain gravity" need a capable model.

    // Simple lookup prefixes
    let fast_prefixes = [
        "what is ",
        "who is ",
        "define ",
        "what's ",
        "who's ",
        "when is ",
        "where is ",
    ];
    for prefix in &fast_prefixes {
        if lower.starts_with(prefix) && word_count <= 6 {
            return ClassificationResult {
                tier: Tier::Fast,
                reason: format!("simple lookup: {}", prefix.trim()),
            };
        }
    }

    // --- Default: Primary ---
    ClassificationResult {
        tier: Tier::Primary,
        reason: "default".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn classify(text: &str) -> Tier {
        classify_query(text).tier
    }

    #[test]
    fn test_smart_code_fence() {
        assert_eq!(
            classify("Here is some code:\n```rust\nfn main() {}\n```"),
            Tier::Smart
        );
    }

    #[test]
    fn test_smart_long_message() {
        let long = "a ".repeat(300); // 600 chars
        assert_eq!(classify(&long), Tier::Smart);
    }

    #[test]
    fn test_smart_multiple_questions() {
        assert_eq!(
            classify("What is this? How does it work? Why is it broken?"),
            Tier::Smart
        );
    }

    #[test]
    fn test_smart_keywords() {
        assert_eq!(classify("implement a web server in Rust"), Tier::Smart);
        assert_eq!(classify("please refactor this module"), Tier::Smart);
        assert_eq!(classify("debug this error for me"), Tier::Smart);
        assert_eq!(classify("analyze the performance"), Tier::Smart);
        assert_eq!(classify("explain this step by step"), Tier::Smart);
        assert_eq!(classify("write code for sorting"), Tier::Smart);
        assert_eq!(classify("optimize the database queries"), Tier::Smart);
        assert_eq!(classify("design a REST API"), Tier::Smart);
    }

    #[test]
    fn test_fast_greetings() {
        assert_eq!(classify("hi"), Tier::Fast);
        assert_eq!(classify("Hello"), Tier::Fast);
        assert_eq!(classify("thanks"), Tier::Fast);
        assert_eq!(classify("ok"), Tier::Fast);
        assert_eq!(classify("yes"), Tier::Fast);
        assert_eq!(classify("no"), Tier::Fast);
        assert_eq!(classify("  hey  "), Tier::Fast);
    }

    #[test]
    fn test_fast_single_word() {
        assert_eq!(classify("test"), Tier::Fast);
        assert_eq!(classify("status"), Tier::Fast);
    }

    #[test]
    fn test_short_message_uses_primary() {
        // Short messages go to primary — short ≠ simple
        assert_eq!(classify("how are you"), Tier::Primary);
        assert_eq!(classify("what time"), Tier::Primary);
        assert_eq!(classify("capital Ecuador"), Tier::Primary);
    }

    #[test]
    fn test_fast_simple_lookup() {
        assert_eq!(classify("what is rust"), Tier::Fast);
        assert_eq!(classify("who is linus torvalds"), Tier::Fast);
        assert_eq!(classify("define recursion"), Tier::Fast);
    }

    #[test]
    fn test_primary_default() {
        assert_eq!(
            classify("Tell me about the history of computing and its impact on modern society"),
            Tier::Primary
        );
        assert_eq!(
            classify("Can you help me with my project setup"),
            Tier::Primary
        );
    }

    #[test]
    fn test_is_uniform_true() {
        let models = ModelsConfig {
            primary: "gpt-4o".to_string(),
            fast: "gpt-4o".to_string(),
            smart: "gpt-4o".to_string(),
        };
        let router = Router::new(models);
        assert!(router.is_uniform());
    }

    #[test]
    fn test_is_uniform_false() {
        let models = ModelsConfig {
            primary: "gpt-4o".to_string(),
            fast: "gpt-4o-mini".to_string(),
            smart: "gpt-4o".to_string(),
        };
        let router = Router::new(models);
        assert!(!router.is_uniform());
    }

    #[test]
    fn test_display_tier() {
        assert_eq!(Tier::Fast.to_string(), "fast");
        assert_eq!(Tier::Primary.to_string(), "primary");
        assert_eq!(Tier::Smart.to_string(), "smart");
    }
}
