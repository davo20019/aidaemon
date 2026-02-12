use std::fmt;

use crate::config::ModelsConfig;
use crate::execution_policy::ModelProfile;

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

    /// Thin profile-to-model mapping used by policy-driven routing.
    /// Cheap -> fast, Balanced -> primary, Strong -> smart.
    pub fn select_for_profile(&self, profile: ModelProfile) -> &str {
        match profile {
            ModelProfile::Cheap => &self.models.fast,
            ModelProfile::Balanced => &self.models.primary,
            ModelProfile::Strong => &self.models.smart,
        }
    }

    /// Returns true when all three tier models resolve to the same string,
    /// meaning auto-routing would be pointless.
    pub fn is_uniform(&self) -> bool {
        self.models.fast == self.models.primary && self.models.primary == self.models.smart
    }
}

/// Classify a user query into a model tier using structural signals only.
pub fn classify_query(text: &str) -> ClassificationResult {
    let trimmed = text.trim();

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

    // --- Fast tier checks ---

    // Single-word messages
    let word_count = trimmed.split_whitespace().count();
    if word_count == 1 {
        return ClassificationResult {
            tier: Tier::Fast,
            reason: "single word".to_string(),
        };
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
    fn test_keywords_no_longer_force_smart() {
        assert_eq!(classify("implement a web server in Rust"), Tier::Primary);
        assert_eq!(classify("please refactor this module"), Tier::Primary);
        assert_eq!(classify("debug this error for me"), Tier::Primary);
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
    fn test_lookup_phrases_use_primary_without_keyword_rules() {
        assert_eq!(classify("what is rust"), Tier::Primary);
        assert_eq!(classify("who is linus torvalds"), Tier::Primary);
        assert_eq!(classify("define recursion"), Tier::Primary);
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

    #[test]
    fn test_select_for_profile() {
        let models = ModelsConfig {
            primary: "primary-model".to_string(),
            fast: "fast-model".to_string(),
            smart: "smart-model".to_string(),
        };
        let router = Router::new(models);
        assert_eq!(router.select_for_profile(ModelProfile::Cheap), "fast-model");
        assert_eq!(
            router.select_for_profile(ModelProfile::Balanced),
            "primary-model"
        );
        assert_eq!(
            router.select_for_profile(ModelProfile::Strong),
            "smart-model"
        );
    }
}
