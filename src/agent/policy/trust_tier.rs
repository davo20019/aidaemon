//! Model trust-tier classification.
//!
//! The agent loop carries two kinds of scaffolding: hard safety caps
//! (iteration/token/wall-clock limits, repetition guards) and behavioral
//! supervision (deferred-action blocking, pre-execution planning/critique,
//! uncertainty gates, plain-text drift redirects). The supervision layer
//! rescues small local models but taxes frontier models with extra
//! round-trips and false-positive interceptions.
//!
//! `ModelTrustTier` makes that a switch: `Guided` models keep the full
//! supervision harness, `Autonomous` models get a thin loop where
//! supervision gates record telemetry instead of blocking. Hard safety
//! caps apply to both tiers.

/// How much the harness trusts the active model to self-direct.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelTrustTier {
    /// Small/local/unknown models: full supervision scaffolding enforced.
    Guided,
    /// Frontier models: supervision gates are shadow-mode (telemetry only).
    Autonomous,
}

impl ModelTrustTier {
    pub fn as_str(self) -> &'static str {
        match self {
            ModelTrustTier::Guided => "guided",
            ModelTrustTier::Autonomous => "autonomous",
        }
    }
}

/// Operator override from config (`[policy] trust_tier = "auto" | "guided" | "autonomous"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TrustTierSetting {
    /// Classify by model id heuristic (default).
    #[default]
    Auto,
    /// Force full supervision regardless of model.
    Guided,
    /// Force thin loop regardless of model.
    Autonomous,
}

impl TrustTierSetting {
    /// Parse the config string. Unknown values fall back to `Auto`.
    pub fn parse(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "guided" => Self::Guided,
            "autonomous" => Self::Autonomous,
            _ => Self::Auto,
        }
    }
}

/// Model-id substrings that mark frontier families. Matched against the
/// lowercased id so provider prefixes ("anthropic/", "us.anthropic.") and
/// version suffixes pass through.
const AUTONOMOUS_FAMILY_MARKERS: &[&str] = &[
    "claude-",
    "gpt-4",
    "gpt-5",
    "gemini-2.5",
    "gemini-3",
    "grok-3",
    "grok-4",
];

/// Classify the active model into a trust tier, honoring the config override.
pub fn classify_model_trust(model: &str, setting: TrustTierSetting) -> ModelTrustTier {
    match setting {
        TrustTierSetting::Guided => return ModelTrustTier::Guided,
        TrustTierSetting::Autonomous => return ModelTrustTier::Autonomous,
        TrustTierSetting::Auto => {}
    }
    let id = model.to_ascii_lowercase();
    if AUTONOMOUS_FAMILY_MARKERS.iter().any(|f| id.contains(f)) {
        return ModelTrustTier::Autonomous;
    }
    // OpenAI o-series ids are too short for substring matching: require the
    // family token to start a path segment ("o3", "o4-mini", "openai/o3"),
    // so names like "yi-large-o3000" stay Guided.
    let o_series = id.split(['/', ':', '.']).any(|seg| {
        ["o1", "o3", "o4"].iter().any(|family| {
            seg.strip_prefix(family)
                .is_some_and(|rest| rest.is_empty() || rest.starts_with('-'))
        })
    });
    if o_series {
        return ModelTrustTier::Autonomous;
    }
    ModelTrustTier::Guided
}

impl crate::agent::Agent {
    /// Trust tier for the model handling the current call, combining the
    /// `[policy] trust_tier` config override with the model-id heuristic.
    pub(crate) fn trust_tier_for_model(&self, model: &str) -> ModelTrustTier {
        classify_model_trust(
            model,
            TrustTierSetting::parse(&self.policy_config.trust_tier),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn auto(model: &str) -> ModelTrustTier {
        classify_model_trust(model, TrustTierSetting::Auto)
    }

    #[test]
    fn frontier_models_classify_autonomous() {
        for model in [
            "claude-fable-5",
            "claude-opus-4-8",
            "claude-sonnet-4-6",
            "anthropic/claude-haiku-4-5-20251001",
            "gpt-5",
            "openai/gpt-4.1",
            "gpt-4o",
            "o3",
            "o4-mini",
            "openrouter/openai/o3",
            "gemini-2.5-pro",
            "gemini-3-pro-preview",
            "grok-4",
            "xai/grok-3-mini",
        ] {
            assert_eq!(
                auto(model),
                ModelTrustTier::Autonomous,
                "expected {model} to classify Autonomous"
            );
        }
    }

    #[test]
    fn small_or_local_models_classify_guided() {
        for model in [
            "gemma-3-27b-it",
            "google/gemma-4-12b",
            "llama-3.1-8b-instruct",
            "qwen2.5-7b-instruct",
            "mistral-small-latest",
            "phi-4",
            "kimi-k2.5",
            "moonshotai/kimi-k2",
            "my-local-gguf-model",
            "",
        ] {
            assert_eq!(
                auto(model),
                ModelTrustTier::Guided,
                "expected {model} to classify Guided"
            );
        }
    }

    #[test]
    fn provider_prefixes_and_casing_are_ignored() {
        assert_eq!(
            auto("US.Anthropic.Claude-Opus-4-8-v1:0"),
            ModelTrustTier::Autonomous
        );
        assert_eq!(auto("OpenAI/GPT-5-Codex"), ModelTrustTier::Autonomous);
    }

    #[test]
    fn o_series_requires_segment_boundary() {
        // "o3"/"o4" must not match inside unrelated names.
        assert_eq!(auto("yi-large-o3000"), ModelTrustTier::Guided);
        assert_eq!(auto("llava-onevision-o4b"), ModelTrustTier::Guided);
    }

    #[test]
    fn config_override_wins_over_heuristic() {
        assert_eq!(
            classify_model_trust("claude-opus-4-8", TrustTierSetting::Guided),
            ModelTrustTier::Guided
        );
        assert_eq!(
            classify_model_trust("gemma-3-27b-it", TrustTierSetting::Autonomous),
            ModelTrustTier::Autonomous
        );
    }

    #[test]
    fn setting_parses_config_strings() {
        assert_eq!(TrustTierSetting::parse("auto"), TrustTierSetting::Auto);
        assert_eq!(TrustTierSetting::parse("Guided"), TrustTierSetting::Guided);
        assert_eq!(
            TrustTierSetting::parse(" AUTONOMOUS "),
            TrustTierSetting::Autonomous
        );
        assert_eq!(
            TrustTierSetting::parse("garbage"),
            TrustTierSetting::Auto,
            "unknown values fall back to Auto"
        );
    }
}
